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
    def __init__(self, d_model, n_nodes, memory_size=1196, k_retrieve=8, tau_time=2.0, tau_contrast=0.07):
        super().__init__()
        self.k = k_retrieve
        self.tau_time = tau_time
        self.tau_contrast = tau_contrast

        self.encoder = TemporalEncoder(d_input=1, d_model=d_model, n_nodes=n_nodes)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.proj = nn.Linear(d_model, d_model)

        self.register_buffer('memory_bank', torch.randn(memory_size, n_nodes, d_model))
        self.register_buffer('memory_seasons', torch.zeros(memory_size, dtype=torch.long))
        self.register_buffer('memory_years', torch.zeros(memory_size))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update_memory(self, new_mem, season_label, year_label):
        ptr = int(self.memory_ptr.item())
        self.memory_bank[ptr] = new_mem.mean(0)
        if season_label.numel() > 0:
            self.memory_seasons[ptr] = int(torch.mode(season_label).values.item())
        if year_label.numel() > 0:
            self.memory_years[ptr] = float(year_label.mean().item())
        self.memory_ptr[0] = (ptr + 1) % self.memory_bank.shape[0]

    def retrieve(self, q, season_q, year_q):
        b, n, d = q.shape
        m = self.memory_bank.shape[0]

        q_flat = F.normalize(q.reshape(b, -1), dim=-1)
        mem_flat = F.normalize(self.memory_bank.reshape(m, -1), dim=-1)
        sim = q_flat @ mem_flat.T

        season_mask = (season_q.unsqueeze(1) == self.memory_seasons.unsqueeze(0)).float()
        sim = sim * season_mask + (1 - season_mask) * (-1e4)

        delta_year = (year_q.unsqueeze(1) - self.memory_years.unsqueeze(0)).abs()
        diversity = 1.0 - torch.exp(-delta_year / max(self.tau_time, 1e-4))
        sim = sim * (0.5 + 0.5 * diversity)

        topk_idx = sim.topk(min(self.k, m), dim=-1).indices
        return self.memory_bank[topk_idx]

    def forward(self, x_scalar, season_q, year_q):
        x4 = x_scalar.unsqueeze(-1)  # [B,T,N,1]
        b, t, n, _ = x4.shape
        q = self.encoder(x4)
        retrieved = self.retrieve(q, season_q, year_q)  # [B,K,N,d]

        q_attn = q.reshape(b * n, 1, -1)
        kv_attn = retrieved.permute(0, 2, 1, 3).reshape(b * n, retrieved.shape[1], -1)
        out, _ = self.cross_attn(q_attn, kv_attn, kv_attn)
        out = out.reshape(b, n, -1)
        return self.proj(out), q

    def contrastive_loss(self, q, season_q, year_q):
        b, n, d = q.shape
        q_flat = F.normalize(q.reshape(b, -1), dim=-1)
        m = self.memory_bank.shape[0]
        mem_flat = F.normalize(self.memory_bank.reshape(m, -1), dim=-1)
        logits = (q_flat @ mem_flat.T) / max(self.tau_contrast, 1e-6)

        delta_year = (year_q.unsqueeze(1) - self.memory_years.unsqueeze(0)).abs()
        season_match = (season_q.unsqueeze(1) == self.memory_seasons.unsqueeze(0))
        pos_mask = season_match & (delta_year > 1.0)

        loss = logits.new_tensor(0.0)
        valid = 0
        for i in range(b):
            pos = pos_mask[i].nonzero(as_tuple=True)[0]
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
        x_last = x_scalar[:, -1, :].unsqueeze(-1)
        z0 = self.encoder(x_last)
        gap_steps = float(gap_years) * 365 * 24 * 12
        z_gap = self.euler_integrate(z0, 0, gap_steps) if gap_steps > 0 else z0
        return self.decoder(z_gap)

    def ode_reconstruction_loss(self, x_scalar):
        b, t, n = x_scalar.shape
        half = max(1, t // 2)
        z0 = self.encoder(x_scalar[:, half - 1, :].unsqueeze(-1))
        z_hat = self.euler_integrate(z0, 0, half)
        z_true = self.encoder(x_scalar[:, -1, :].unsqueeze(-1).detach())
        loss_ode = F.mse_loss(z_hat, z_true)
        z_mid = self.euler_integrate(z0, 0, max(1, half // 2))
        loss_smooth = (z_hat - 2 * z_mid + z0).pow(2).mean()
        return loss_ode, loss_smooth


class AnchorTemporalFusion(nn.Module):
    def __init__(self, d_model, n_gap_types=4):
        super().__init__()
        self.gap_embed = nn.Embedding(n_gap_types, d_model)
        self.x_proj = nn.Linear(1, d_model)
        self.gate_net = nn.Sequential(nn.Linear(d_model * 4, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, 3))
        self.out_proj = nn.Linear(d_model, d_model)
        self.delta_head = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)

    def gap_to_idx(self, gap_years):
        mapping = {0.0: 0, 1.0: 1, 1.5: 2, 2.0: 3}
        return mapping.get(float(gap_years), 0)

    def forward(self, h_cem, h_lde, x_scalar, gap_years=0.0):
        b, t, n = x_scalar.shape
        x_enc_embed = self.x_proj(x_scalar[:, -1, :].unsqueeze(-1))

        gap_idx = self.gap_to_idx(gap_years)
        gap_ids = torch.full((b,), gap_idx, dtype=torch.long, device=x_scalar.device)
        e_gap = self.gap_embed(gap_ids).unsqueeze(1).expand(-1, n, -1)

        gate_in = torch.cat([h_cem, h_lde, x_enc_embed, e_gap], dim=-1)
        gates = F.softmax(self.gate_net(gate_in), dim=-1)
        z = gates[..., 0:1] * h_cem + gates[..., 1:2] * h_lde + gates[..., 2:3] * x_enc_embed
        z = self.norm(self.out_proj(z))

        delta = self.delta_head(z).squeeze(-1)
        return x_scalar + delta.unsqueeze(1).expand(-1, t, -1)


class Model(nn.Module):
    """CAMEL standalone forecasting model, parallel to DLinear/Autoformer."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len if self.task_name in ['long_term_forecast', 'short_term_forecast'] else configs.seq_len
        self.n_nodes = configs.enc_in

        self.camel_gap_years = getattr(configs, 'camel_gap_years', 0.0)
        self.lambda_mem = getattr(configs, 'lambda_mem', 0.10)
        self.lambda_ode = getattr(configs, 'lambda_ode', 0.05)
        self.lambda_smooth = getattr(configs, 'lambda_smooth', 0.01)

        d_model = getattr(configs, 'camel_d_model', 32)
        d_latent = getattr(configs, 'camel_latent_dim', 32)
        memory_size = getattr(configs, 'camel_memory_size', 1196)
        k_retrieve = getattr(configs, 'camel_k_retrieve', 8)

        self.cem = CrossYearEpisodicMemory(d_model=d_model, n_nodes=self.n_nodes,
                                           memory_size=memory_size, k_retrieve=k_retrieve)
        self.lde = LatentDynamicsExtrapolator(d_model=d_model, n_nodes=self.n_nodes, d_latent=d_latent)
        self.atf = AnchorTemporalFusion(d_model=d_model)

        self.temporal_head = nn.Linear(self.seq_len, self.pred_len)
        self.temporal_head.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def _extract_meta(self, x_mark_enc, x_enc):
        b = x_enc.shape[0]
        season_q = torch.zeros(b, dtype=torch.long, device=x_enc.device)
        year_q = torch.zeros(b, dtype=x_enc.dtype, device=x_enc.device)

        if x_mark_enc is not None and x_mark_enc.shape[-1] > 0:
            month_like = x_mark_enc[:, -1, 0]
            if month_like.min() >= 1 and month_like.max() <= 12:
                season_q = ((month_like.long() - 1) // 3).clamp(min=0, max=3)

            if x_mark_enc.shape[-1] >= 5:
                year_q = x_mark_enc[:, -1, -1].float().clamp(0.0, 1.0)
        return season_q, year_q

    def _camel_enhance(self, x_enc, x_mark_enc):
        season_q, year_q = self._extract_meta(x_mark_enc, x_enc)

        h_cem, q = self.cem(x_enc, season_q, year_q)
        h_lde = self.lde(x_enc, self.camel_gap_years)
        x_enh = self.atf(h_cem, h_lde, x_enc, self.camel_gap_years)

        loss_mem = self.cem.contrastive_loss(q, season_q, year_q)
        loss_ode, loss_smooth = self.lde.ode_reconstruction_loss(x_enc)
        aux_loss = self.lambda_mem * loss_mem + self.lambda_ode * loss_ode + self.lambda_smooth * loss_smooth

        if self.training:
            with torch.no_grad():
                self.cem.update_memory(q.detach(), season_q.detach(), year_q.detach())

        return x_enh, aux_loss

    def forecast(self, x_enc, x_mark_enc):
        x_enh, aux_loss = self._camel_enhance(x_enc, x_mark_enc)
        out = self.temporal_head(x_enh.transpose(1, 2)).transpose(1, 2)
        return out, aux_loss

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name not in ['long_term_forecast', 'short_term_forecast']:
            out, _ = self.forecast(x_enc, x_mark_enc)
            return out

        out, aux_loss = self.forecast(x_enc, x_mark_enc)
        # trick: keep return shape aligned, inject aux loss into graph without changing outputs.
        out = out + 0.0 * aux_loss
        return out[:, -self.pred_len:, :]
