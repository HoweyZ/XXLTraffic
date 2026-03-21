import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    def __init__(self, d_input, d_model, n_nodes):
        super().__init__()
        in_ch = n_nodes * d_input
        hid_ch = n_nodes * d_model
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size=12, groups=n_nodes, padding=6)
        self.pointwise = nn.Conv1d(in_ch, hid_ch, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(d_model)
        self.n_nodes = n_nodes

    def forward(self, x):
        # x: [B, T, N, d_input]
        b, t, n, d = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, n * d, t)
        x = F.gelu(self.depthwise(x))
        x = F.gelu(self.pointwise(x))
        x = self.pool(x).squeeze(-1).reshape(b, self.n_nodes, -1)
        return self.norm(x)


class CrossYearEpisodicMemory(nn.Module):
    def __init__(self, d_model, n_nodes, memory_size=1196, k_retrieve=8, tau_time=2.0, tau_contrast=0.07, min_year_gap=1.0, gap_tolerance=0.25, debug_stats=False, debug_interval=100):
        super().__init__()
        self.k = k_retrieve
        self.tau_time = tau_time
        self.tau_contrast = tau_contrast
        self.min_year_gap = float(min_year_gap)
        self.gap_tolerance = float(gap_tolerance)
        self.debug_stats = bool(debug_stats)
        self.debug_interval = max(1, int(debug_interval))

        self.encoder = TemporalEncoder(d_input=1, d_model=d_model, n_nodes=n_nodes)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.proj = nn.Linear(d_model, d_model)

        self.register_buffer('memory_bank', torch.randn(memory_size, n_nodes, d_model))
        self.register_buffer('memory_seasons', torch.zeros(memory_size, dtype=torch.long))
        self.register_buffer('memory_years', torch.zeros(memory_size))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('memory_count', torch.zeros(1, dtype=torch.long))
        self.register_buffer('debug_counter', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update_memory(self, new_mem, season_label, year_label):
        ptr = int(self.memory_ptr.item())
        self.memory_bank[ptr] = new_mem.mean(0)
        if season_label.numel() > 0:
            self.memory_seasons[ptr] = int(torch.mode(season_label).values.item())
        if year_label.numel() > 0:
            self.memory_years[ptr] = float(year_label.mean().item())
        self.memory_ptr[0] = (ptr + 1) % self.memory_bank.shape[0]
        self.memory_count[0] = min(int(self.memory_count.item()) + 1, self.memory_bank.shape[0])

    def retrieve(self, q, season_q, year_q, gap_years=0.0):
        b, _, _ = q.shape
        m = int(self.memory_count.item())
        if m <= 0:
            return q.unsqueeze(1)

        q_flat = F.normalize(q.reshape(b, -1), dim=-1)
        memory_bank = self.memory_bank[:m]
        memory_seasons = self.memory_seasons[:m]
        memory_years = self.memory_years[:m]

        mem_flat = F.normalize(memory_bank.reshape(m, -1), dim=-1)
        sim = q_flat @ mem_flat.T

        season_mask = (season_q.unsqueeze(1) == memory_seasons.unsqueeze(0)).float()
        sim = sim * season_mask + (1 - season_mask) * (-1e4)

        delta_year = (year_q.unsqueeze(1) - memory_years.unsqueeze(0)).abs()
        if gap_years and gap_years > 0:
            gap_center = torch.as_tensor(float(gap_years), dtype=delta_year.dtype, device=delta_year.device)
            tol = max(self.gap_tolerance, 1e-4)
            gap_weight = torch.exp(-((delta_year - gap_center) ** 2) / (2 * tol * tol))
            sim = sim + torch.log(gap_weight + 1e-6)
        else:
            diversity = 1.0 - torch.exp(-delta_year / max(self.tau_time, 1e-4))
            sim = sim * (0.5 + 0.5 * diversity)

        topk_idx = sim.topk(min(self.k, m), dim=-1).indices
        return memory_bank[topk_idx]

    def forward(self, x_scalar, season_q, year_q, gap_years=0.0):
        # x_scalar: [B, T, N]
        q = self.encoder(x_scalar.unsqueeze(-1))
        retrieved = self.retrieve(q, season_q, year_q, gap_years=gap_years)  # [B,K,N,d]

        b, n, _ = q.shape
        q_attn = q.reshape(b * n, 1, -1)
        kv_attn = retrieved.permute(0, 2, 1, 3).reshape(b * n, retrieved.shape[1], -1)
        out, _ = self.cross_attn(q_attn, kv_attn, kv_attn)
        out = out.reshape(b, n, -1)
        return self.proj(out), q

    def contrastive_loss(self, q, season_q, year_q, gap_years=0.0, log_stats=False):
        b, _, _ = q.shape
        m = int(self.memory_count.item())
        if m <= 0:
            return q.new_tensor(0.0)

        memory_bank = self.memory_bank[:m]
        memory_seasons = self.memory_seasons[:m]
        memory_years = self.memory_years[:m]

        q_flat = F.normalize(q.reshape(b, -1), dim=-1)
        mem_flat = F.normalize(memory_bank.reshape(m, -1), dim=-1)
        logits = (q_flat @ mem_flat.T) / max(self.tau_contrast, 1e-6)

        delta_year = (year_q.unsqueeze(1) - memory_years.unsqueeze(0)).abs()
        season_match = season_q.unsqueeze(1) == memory_seasons.unsqueeze(0)

        if gap_years and gap_years > 0:
            gap_center = torch.as_tensor(float(gap_years), dtype=delta_year.dtype, device=delta_year.device)
            tol = max(self.gap_tolerance, 1e-4)
            pos_mask = season_match & ((delta_year - gap_center).abs() <= tol)
        else:
            pos_mask = season_match & (delta_year > self.min_year_gap)

        if self.debug_stats and log_stats:
            step = int(self.debug_counter.item())
            if step % self.debug_interval == 0:
                season_unique, season_counts = torch.unique(season_q.detach().cpu(), return_counts=True)
                season_dist = {int(k.item()): int(v.item()) for k, v in zip(season_unique, season_counts)}
                hit_rate = float(pos_mask.any(dim=1).float().mean().item())
                print(f"[CAMEL DEBUG] batch={step} season_q_dist={season_dist} pos_mask_hit_rate={hit_rate:.4f}")
            self.debug_counter[0] = step + 1

        loss = logits.new_tensor(0.0)
        valid = 0
        for i in range(b):
            pos = pos_mask[i].nonzero(as_tuple=True)[0]
            if pos.numel() == 0:
                fallback = (season_match[i] & (delta_year[i] > self.min_year_gap)).nonzero(as_tuple=True)[0]
                pos = fallback
            if pos.numel() == 0:
                continue
            pos_logit = logits[i, pos].mean()
            loss = loss + (-pos_logit + torch.logsumexp(logits[i], dim=0))
            valid += 1
        return loss / max(valid, 1)


class TimeEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.periods = [1, 288, 2016, 105120]
        self.proj = nn.Linear(len(self.periods) * 2, d_model)

    def forward(self, t_scalar):
        feats = []
        for p in self.periods:
            feats.append(torch.cos(2 * torch.pi * t_scalar / p))
            feats.append(torch.sin(2 * torch.pi * t_scalar / p))
        return self.proj(torch.stack(feats, dim=-1))


class GraphODEFunc(nn.Module):
    def __init__(self, d_latent, n_nodes, d_embed=32):
        super().__init__()
        self.n_nodes = n_nodes
        self.node_emb1 = nn.Embedding(n_nodes, d_embed)
        self.node_emb2 = nn.Embedding(n_nodes, d_embed)
        self.time_mod = nn.Sequential(nn.Linear(d_embed, d_embed), nn.GELU(), nn.Linear(d_embed, n_nodes * n_nodes))
        self.graph_proj = nn.Linear(d_latent, d_latent)
        self.time_mlp = nn.Sequential(nn.Linear(d_latent + d_embed, d_latent * 2), nn.GELU(), nn.Linear(d_latent * 2, d_latent))
        self.time_enc = TimeEncoding(d_embed)
        self.norm = nn.LayerNorm(d_latent)

    def adaptive_adj(self, t_feat):
        idx = torch.arange(self.n_nodes, device=t_feat.device)
        e1 = self.node_emb1(idx)
        e2 = self.node_emb2(idx)
        base_adj = e1 @ e2.T
        mod = self.time_mod(t_feat).reshape(-1, self.n_nodes, self.n_nodes)
        return F.softmax(base_adj.unsqueeze(0) * mod, dim=-1)

    def forward(self, t, z):
        b = z.shape[0]
        t_tensor = torch.full((b,), float(t), device=z.device, dtype=z.dtype)
        t_feat = self.time_enc(t_tensor)
        adj = self.adaptive_adj(t_feat)
        z_spatial = F.gelu(self.graph_proj(torch.bmm(adj, z)))
        t_expand = t_feat.unsqueeze(1).expand(-1, self.n_nodes, -1)
        z_time = self.time_mlp(torch.cat([z, t_expand], dim=-1))
        return self.norm(z_spatial + z_time)


class LatentDynamicsExtrapolator(nn.Module):
    def __init__(self, d_model, n_nodes, d_latent=32, n_euler_steps=12):
        super().__init__()
        self.n_euler_steps = n_euler_steps
        self.encoder = nn.Sequential(nn.Linear(1, d_latent * 2), nn.GELU(), nn.Linear(d_latent * 2, d_latent))
        self.ode_func = GraphODEFunc(d_latent, n_nodes)
        self.decoder = nn.Sequential(nn.Linear(d_latent, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))

    def euler_integrate(self, z0, t_start, t_end):
        dt = (t_end - t_start) / max(self.n_euler_steps, 1)
        z = z0
        t = float(t_start)
        for _ in range(self.n_euler_steps):
            z = z + dt * self.ode_func(t, z)
            t += dt
        return z

    def forward(self, x_scalar, gap_years=0.0):
        z0 = self.encoder(x_scalar[:, -1, :].unsqueeze(-1))
        gap_steps = float(gap_years) * 365 * 24 * 12
        z_gap = self.euler_integrate(z0, 0, gap_steps) if gap_steps > 0 else z0
        return self.decoder(z_gap)

    def ode_reconstruction_loss(self, x_scalar):
        _, t, _ = x_scalar.shape
        half = max(1, t // 2)
        z0 = self.encoder(x_scalar[:, half - 1, :].unsqueeze(-1))
        z_hat = self.euler_integrate(z0, 0, half)
        z_true = self.encoder(x_scalar[:, -1, :].unsqueeze(-1).detach())
        loss_ode = F.mse_loss(z_hat, z_true)
        z_mid = self.euler_integrate(z0, 0, max(1, half // 2))
        loss_smooth = (z_hat - 2 * z_mid + z0).pow(2).mean()
        return loss_ode, loss_smooth


class UncertaintyHead(nn.Module):
    def __init__(self, d_model, horizon):
        super().__init__()
        hid = max(4, d_model // 2)
        self.net = nn.Sequential(nn.Linear(d_model, hid), nn.GELU(), nn.Linear(hid, horizon), nn.Softplus())

    def forward(self, z):
        return self.net(z) + 1e-6


class AnchorTemporalFusion(nn.Module):
    def __init__(self, d_model, horizon, n_gap_types=4):
        super().__init__()
        self.gap_embed = nn.Embedding(n_gap_types, d_model)
        self.x_proj = nn.Linear(1, d_model)
        self.gate_net = nn.Sequential(nn.Linear(d_model * 4, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, 3))
        self.out_proj = nn.Linear(d_model, d_model)
        self.delta_head = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.uncertainty = UncertaintyHead(d_model, horizon)

    def gap_to_idx(self, gap_years):
        return {0.0: 0, 1.0: 1, 1.5: 2, 2.0: 3}.get(float(gap_years), 0)

    def forward(self, h_cem, h_lde, x_scalar, gap_years=0.0):
        b, t, n = x_scalar.shape
        x_enc_embed = self.x_proj(x_scalar[:, -1, :].unsqueeze(-1))

        gap_ids = torch.full((b,), self.gap_to_idx(gap_years), dtype=torch.long, device=x_scalar.device)
        e_gap = self.gap_embed(gap_ids).unsqueeze(1).expand(-1, n, -1)

        gates = F.softmax(self.gate_net(torch.cat([h_cem, h_lde, x_enc_embed, e_gap], dim=-1)), dim=-1)
        z = gates[..., 0:1] * h_cem + gates[..., 1:2] * h_lde + gates[..., 2:3] * x_enc_embed
        z = self.norm(self.out_proj(z))

        sigma = self.uncertainty(z)
        delta = self.delta_head(z).squeeze(-1)
        z_out = x_scalar + delta.unsqueeze(1).expand(-1, t, -1)
        return z_out, sigma


class NLLLoss(nn.Module):
    def forward(self, pred, target, sigma):
        return (((pred - target) ** 2) / (2 * sigma ** 2) + torch.log(sigma)).mean()


class CAMELCore(nn.Module):
    def __init__(self, d_model, n_nodes, memory_size=1196, k_retrieve=8, d_latent=32, horizon=12, min_year_gap=1.0, gap_tolerance=0.25, debug_stats=False, debug_interval=100):
        super().__init__()
        self.cem = CrossYearEpisodicMemory(d_model=d_model, n_nodes=n_nodes, memory_size=memory_size, k_retrieve=k_retrieve, min_year_gap=min_year_gap, gap_tolerance=gap_tolerance, debug_stats=debug_stats, debug_interval=debug_interval)
        self.lde = LatentDynamicsExtrapolator(d_model=d_model, n_nodes=n_nodes, d_latent=d_latent)
        self.atf = AnchorTemporalFusion(d_model=d_model, horizon=horizon)

    def forward(self, x_scalar, season_q, year_q, gap_years=0.0, update_memory=True):
        h_cem, q = self.cem(x_scalar, season_q, year_q, gap_years=gap_years)
        h_lde = self.lde(x_scalar, gap_years)
        z_out, sigma = self.atf(h_cem, h_lde, x_scalar, gap_years)

        loss_mem = self.cem.contrastive_loss(q, season_q, year_q, gap_years=gap_years, log_stats=update_memory)
        loss_ode, loss_smooth = self.lde.ode_reconstruction_loss(x_scalar)

        if update_memory:
            with torch.no_grad():
                self.cem.update_memory(q.detach(), season_q.detach(), year_q.detach())

        aux = {'mem': loss_mem, 'ode': loss_ode, 'smooth': loss_smooth}
        return z_out, sigma, aux


class Model(nn.Module):
    """CAMEL as a standalone forecasting model parallel to DLinear/Autoformer."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len if self.task_name in ['long_term_forecast', 'short_term_forecast'] else configs.seq_len
        self.output_attention = getattr(configs, 'output_attention', False)

        self.core = CAMELCore(
            d_model=configs.camel_d_model,
            n_nodes=configs.enc_in,
            memory_size=configs.camel_memory_size,
            k_retrieve=configs.camel_k_retrieve,
            d_latent=configs.camel_latent_dim,
            horizon=self.pred_len,
            min_year_gap=getattr(configs, 'camel_min_year_gap', 1.0),
            gap_tolerance=getattr(configs, 'camel_gap_tolerance', 0.25),
            debug_stats=getattr(configs, 'camel_debug_stats', False),
            debug_interval=getattr(configs, 'camel_debug_interval', 100),
        )
        self.temporal_proj = nn.Linear(self.seq_len, self.pred_len)
        self.camel_gap_years = getattr(configs, 'camel_gap_years', 0.0)

    def _extract_meta(self, x_mark_enc, x_enc):
        b = x_enc.shape[0]
        season_q = torch.zeros(b, dtype=torch.long, device=x_enc.device)
        year_q = torch.zeros(b, dtype=x_enc.dtype, device=x_enc.device)

        if x_mark_enc is None or x_mark_enc.shape[-1] == 0:
            return season_q, year_q

        month_like = x_mark_enc[:, -1, 0]
        if month_like.min() >= 1 and month_like.max() <= 12:
            season_q = ((month_like.long() - 1) // 3).clamp(min=0, max=3)

        if x_mark_enc.shape[-1] >= 5:
            year_q = x_mark_enc[:, -1, -1].float()

        return season_q, year_q

    def _forward_forecast(self, x_enc, x_mark_enc):
        season_q, year_q = self._extract_meta(x_mark_enc, x_enc)
        enhanced, sigma, aux_losses = self.core(
            x_enc,
            season_q=season_q,
            year_q=year_q,
            gap_years=getattr(self, 'camel_gap_years', 0.0),
            update_memory=self.training,
        )
        pred = self.temporal_proj(enhanced.transpose(1, 2)).transpose(1, 2)
        return pred[:, -self.pred_len:, :], {'sigma': sigma, 'aux_losses': aux_losses}

    @property
    def camel_gap_years(self):
        return getattr(self, '_camel_gap_years', 0.0)

    @camel_gap_years.setter
    def camel_gap_years(self, value):
        self._camel_gap_years = float(value)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            pred, aux = self._forward_forecast(x_enc, x_mark_enc)
            return pred, aux
        if self.task_name in ['imputation', 'anomaly_detection']:
            pred, _ = self._forward_forecast(x_enc, x_mark_enc)
            return pred
        if self.task_name == 'classification':
            pred, _ = self._forward_forecast(x_enc, x_mark_enc)
            return pred.reshape(pred.shape[0], -1)
        return None
