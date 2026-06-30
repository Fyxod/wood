"""Joint WOOD geometry: TPS + Delaunay + rolling + DCT + FFT phase."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .dct import dct_basis
from .delaunay import delaunay_barycentric
from .fft_phase import FFTPhasePerturbation
from .rolling import rolling_field
from .tps import tps_basis


@dataclass
class WoodGeometryConfig:
    init: str = "neutral"
    init_fraction: float = 0.05
    tps_size: int = 5
    delaunay_size: int = 5
    dct_size: int = 4
    fft_phase_size: int = 8
    edge_falloff_px: float = 16.0
    tps_enabled: bool = True
    delaunay_enabled: bool = True
    rolling_enabled: bool = True
    dct_enabled: bool = True
    fft_phase_enabled: bool = True
    tps_norm_limit: float = 0.007
    delaunay_norm_limit: float = 0.010
    rolling_norm_limit: float = 0.009
    dct_norm_limit: float = 0.008
    tps_px_limit: float | None = None
    delaunay_px_limit: float | None = None
    rolling_px_limit: float | None = None
    dct_px_limit: float | None = None
    fft_phase_limit_rad: float = math.pi
    max_combined_disp_px: float | None = None


def _limit_px(norm_limit: float, height: int, width: int) -> float:
    return float(norm_limit) * float(max(height, width))


def _configured_limit_px(px_limit: float | None, norm_limit: float, height: int, width: int) -> float:
    if px_limit is not None:
        return float(px_limit)
    return _limit_px(norm_limit, height, width)


def load_wood_geometry_config(path: str | Path | None) -> WoodGeometryConfig:
    """Load a JSON geometry config.

    The file may use top-level dataclass keys and/or the friendlier nested
    structure used by `configs/geometry_default.json`.
    """

    if path is None:
        return WoodGeometryConfig()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values: dict[str, Any] = {}
    allowed = set(WoodGeometryConfig.__dataclass_fields__)
    for key, value in payload.items():
        if key in allowed:
            values[key] = value
    for key, value in payload.get("sizes", {}).items():
        if key in allowed:
            values[key] = value
    for key, value in payload.get("global", {}).items():
        if key in allowed:
            values[key] = value
    components = payload.get("components", {})
    mapping = {
        "tps": "tps",
        "delaunay": "delaunay",
        "rolling": "rolling",
        "dct": "dct",
        "fft_phase": "fft_phase",
    }
    for name, prefix in mapping.items():
        block = components.get(name, {})
        if "enabled" in block:
            values[f"{prefix}_enabled"] = bool(block["enabled"])
        if name != "fft_phase":
            if "norm_limit" in block:
                values[f"{prefix}_norm_limit"] = float(block["norm_limit"])
            if "px_limit" in block:
                values[f"{prefix}_px_limit"] = None if block["px_limit"] is None else float(block["px_limit"])
        elif "phase_limit_rad" in block:
            values["fft_phase_limit_rad"] = float(block["phase_limit_rad"])
    return WoodGeometryConfig(**values)


def _field_stats(field: torch.Tensor, prefix: str) -> dict[str, float]:
    mag = torch.sqrt(field.detach().float().square().sum(dim=1) + 1e-12)
    return {
        f"{prefix}_mean_disp": float(mag.mean().cpu()),
        f"{prefix}_max_disp": float(mag.max().cpu()),
        f"{prefix}_p95_disp": float(torch.quantile(mag.flatten(), 0.95).cpu()),
    }


def displacement_stats(field: torch.Tensor) -> dict[str, float]:
    mag = torch.sqrt(field.detach().float().square().sum(dim=1) + 1e-12)
    return {
        "combined_max_disp_px": float(mag.max().cpu()),
        "combined_mean_disp_px": float(mag.mean().cpu()),
        "combined_p95_disp_px": float(torch.quantile(mag.flatten(), 0.95).cpu()),
    }


def smoothness_tv(field: torch.Tensor) -> torch.Tensor:
    return (field[:, :, :, 1:] - field[:, :, :, :-1]).abs().mean() + (
        field[:, :, 1:] - field[:, :, :-1]
    ).abs().mean()


def jacobian_diagnostics(field: torch.Tensor) -> dict[str, float]:
    dx, dy = field[:, 0], field[:, 1]
    ddx = F.pad((dx[:, :, 2:] - dx[:, :, :-2]) / 2.0, (1, 1))
    dxy = F.pad((dx[:, 2:] - dx[:, :-2]) / 2.0, (0, 0, 1, 1))
    dyx = F.pad((dy[:, :, 2:] - dy[:, :, :-2]) / 2.0, (1, 1))
    ddy = F.pad((dy[:, 2:] - dy[:, :-2]) / 2.0, (0, 0, 1, 1))
    det = (1.0 + ddx) * (1.0 + ddy) - dxy * dyx
    return {
        "jacobian_det_min": float(det.detach().float().min().cpu()),
        "foldover_fraction": float((det.detach().float() < 0).float().mean().cpu()),
        "smoothness_tv": float(smoothness_tv(field.detach().float()).cpu()),
    }


class CombinedWoodPerturbation(torch.nn.Module):
    """Combined differentiable WOOD perturbation module.

    Four spatial fields are summed and applied with grid_sample. FFT phase is
    then applied as a differentiable frequency-domain stage.
    """

    def __init__(
        self,
        height: int,
        width: int,
        channels: int,
        device: torch.device,
        seed: int = 1234,
        config: WoodGeometryConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or WoodGeometryConfig()
        self.height = int(height)
        self.width = int(width)
        self.channels = int(channels)
        self.tps_limit_px = _configured_limit_px(self.config.tps_px_limit, self.config.tps_norm_limit, height, width)
        self.delaunay_limit_px = _configured_limit_px(
            self.config.delaunay_px_limit, self.config.delaunay_norm_limit, height, width
        )
        self.rolling_limit_px = _configured_limit_px(self.config.rolling_px_limit, self.config.rolling_norm_limit, height, width)
        self.dct_limit_px = _configured_limit_px(self.config.dct_px_limit, self.config.dct_norm_limit, height, width)
        self.component_limit_for_flow = max(
            self.tps_limit_px,
            self.delaunay_limit_px,
            self.rolling_limit_px,
            self.dct_limit_px,
            1.0,
        )

        generator = torch.Generator(device=device).manual_seed(seed + 9101)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device),
            torch.linspace(-1, 1, width, device=device),
            indexing="ij",
        )
        self.register_buffer("base_grid", torch.stack([xx, yy], dim=-1)[None])
        self.register_buffer("yy", yy[None, None])

        distances = torch.minimum(
            torch.minimum(torch.arange(width, device=device)[None], torch.arange(width - 1, -1, -1, device=device)[None]),
            torch.minimum(torch.arange(height, device=device)[:, None], torch.arange(height - 1, -1, -1, device=device)[:, None]),
        ).float()
        t = (distances / max(float(self.config.edge_falloff_px), 1.0)).clamp(0, 1)
        edge = t * t * (3 - 2 * t)
        self.register_buffer("edge", edge[None, None])

        self.register_buffer("dct_basis", dct_basis(self.config.dct_size, height, width, device))
        self.register_buffer("tps_matrix", tps_basis(self.config.tps_size, height, width, device))
        delaunay_idx, delaunay_weight = delaunay_barycentric(self.config.delaunay_size, height, width, device)
        self.register_buffer("delaunay_idx", delaunay_idx)
        self.register_buffer("delaunay_weight", delaunay_weight)

        def init_tensor(shape, limit: float):
            if self.config.init == "small_random":
                return torch.randn(*shape, device=device, generator=generator) * (limit * self.config.init_fraction)
            return torch.zeros(*shape, device=device)

        self.tps_raw = torch.nn.Parameter(init_tensor((1, 2, self.config.tps_size, self.config.tps_size), self.tps_limit_px))
        self.delaunay_raw = torch.nn.Parameter(
            init_tensor((1, 2, self.config.delaunay_size, self.config.delaunay_size), self.delaunay_limit_px)
        )
        self.dct_coeffs = torch.nn.Parameter(init_tensor((2, self.dct_basis.shape[0]), self.dct_limit_px))
        self.roll_params = torch.nn.Parameter(init_tensor((2,), self.rolling_limit_px))

        tps_mask = torch.ones_like(self.tps_raw)
        tps_mask[:, :, 0] = 0
        tps_mask[:, :, -1] = 0
        tps_mask[:, :, :, 0] = 0
        tps_mask[:, :, :, -1] = 0
        self.register_buffer("tps_mask", tps_mask)
        delaunay_mask = torch.ones_like(self.delaunay_raw)
        delaunay_mask[:, :, 0] = 0
        delaunay_mask[:, :, -1] = 0
        delaunay_mask[:, :, :, 0] = 0
        delaunay_mask[:, :, :, -1] = 0
        self.register_buffer("delaunay_mask", delaunay_mask)

        fft_init = 0.0 if self.config.init == "neutral" else 0.05 * torch.pi
        self.fft_phase = FFTPhasePerturbation(
            channels,
            self.config.fft_phase_size,
            float(fft_init),
            device,
            seed,
            max_phase_rad=self.config.fft_phase_limit_rad,
        )
        self.tps_raw.requires_grad_(self.config.tps_enabled)
        self.delaunay_raw.requires_grad_(self.config.delaunay_enabled)
        self.dct_coeffs.requires_grad_(self.config.dct_enabled)
        self.roll_params.requires_grad_(self.config.rolling_enabled)
        self.fft_phase.raw_phase.requires_grad_(self.config.fft_phase_enabled)
        self.project_()

    def _zero_field(self) -> torch.Tensor:
        return self.base_grid.new_zeros((1, 2, self.height, self.width))

    def _tps_field(self) -> torch.Tensor:
        if not self.config.tps_enabled:
            return self._zero_field()
        controls = (self.tps_raw.clamp(-self.tps_limit_px, self.tps_limit_px) * self.tps_mask).reshape(1, 2, -1)
        field = torch.einsum("pn,bcn->bcp", self.tps_matrix, controls)
        return field.reshape(1, 2, self.height, self.width)

    def _delaunay_field(self) -> torch.Tensor:
        if not self.config.delaunay_enabled:
            return self._zero_field()
        controls = (self.delaunay_raw.clamp(-self.delaunay_limit_px, self.delaunay_limit_px) * self.delaunay_mask).reshape(1, 2, -1)
        gathered = controls[:, :, self.delaunay_idx.flatten()].reshape(1, 2, -1, 3)
        field = (gathered * self.delaunay_weight[None, None]).sum(-1)
        return field.reshape(1, 2, self.height, self.width)

    def _dct_field(self) -> torch.Tensor:
        if not self.config.dct_enabled:
            return self._zero_field()
        coeffs = self.dct_coeffs.clamp(-self.dct_limit_px, self.dct_limit_px)
        return torch.einsum("ck,khw->chw", coeffs, self.dct_basis)[None]

    def _rolling_field(self) -> torch.Tensor:
        if not self.config.rolling_enabled:
            return self._zero_field()
        params = self.roll_params.clamp(-self.rolling_limit_px, self.rolling_limit_px)
        return rolling_field(self.yy, params[0], params[1])

    def spatial_fields(self) -> dict[str, torch.Tensor]:
        return {
            "tps": self._tps_field() * self.edge,
            "delaunay": self._delaunay_field() * self.edge,
            "rolling": self._rolling_field() * self.edge,
            "dct": self._dct_field() * self.edge,
        }

    def spatial_warp(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        fields = self.spatial_fields()
        displacement = sum(fields.values())
        if self.config.max_combined_disp_px is not None and self.config.max_combined_disp_px > 0:
            magnitude = torch.sqrt(displacement.square().sum(dim=1, keepdim=True) + 1e-12)
            cap = float(self.config.max_combined_disp_px)
            displacement = displacement * torch.clamp(cap / magnitude.clamp_min(1e-6), max=1.0)
        grid = self.base_grid.clone()
        grid[..., 0] += 2.0 * displacement[:, 0] / max(self.width - 1, 1)
        grid[..., 1] += 2.0 * displacement[:, 1] / max(self.height - 1, 1)
        warped = F.grid_sample(image, grid, mode="bilinear", padding_mode="reflection", align_corners=True).clamp(0, 1)
        return warped, displacement, fields

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        spatial, displacement, fields = self.spatial_warp(image)
        if self.config.fft_phase_enabled:
            perturbed, fft_delta, fft_stats = self.fft_phase(spatial)
        else:
            perturbed = spatial
            fft_delta = torch.zeros_like(spatial)
            fft_stats = {
                "fft_phase_norm": 0.0,
                "fft_phase_mean_abs": 0.0,
                "fft_phase_max_abs": 0.0,
                "legacy_fft_strength_equivalent": 0.0,
                "fft_spatial_delta_mse": 0.0,
            }
        diagnostics = self.diagnostics(displacement, fields)
        diagnostics.update(fft_stats if isinstance(fft_stats, dict) else fft_stats.__dict__)
        return perturbed, {
            "spatial": spatial,
            "displacement": displacement,
            "fields": fields,
            "fft_delta": fft_delta,
            "diagnostics": diagnostics,
        }

    def diagnostics(self, displacement: torch.Tensor, fields: dict[str, torch.Tensor]) -> dict[str, float]:
        out: dict[str, float] = {}
        out.update(displacement_stats(displacement))
        out.update(jacobian_diagnostics(displacement))
        for name, field in fields.items():
            out.update(_field_stats(field, name))
        return out

    def grad_norms(self) -> dict[str, float]:
        def norm(parameters) -> float:
            values = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
            if not values:
                return 0.0
            return float(torch.stack(values).sum().sqrt().cpu())

        return {
            "tps_grad_norm": norm([self.tps_raw]),
            "delaunay_grad_norm": norm([self.delaunay_raw]),
            "rolling_grad_norm": norm([self.roll_params]),
            "dct_grad_norm": norm([self.dct_coeffs]),
            "fft_phase_grad_norm": norm([self.fft_phase.raw_phase]),
            "total_grad_norm": norm(list(self.parameters())),
        }

    def _param_stats(self, tensor: torch.Tensor, limit: float, prefix: str) -> dict[str, float | int]:
        data = tensor.detach().float()
        return {
            f"{prefix}_param_min": float(data.min().cpu()),
            f"{prefix}_param_max": float(data.max().cpu()),
            f"{prefix}_param_mean_abs": float(data.abs().mean().cpu()),
            f"{prefix}_num_at_min": int((data <= -limit + 1e-8).sum().cpu()),
            f"{prefix}_num_at_max": int((data >= limit - 1e-8).sum().cpu()),
        }

    def parameter_diagnostics(self) -> dict[str, float | int | str]:
        stats: dict[str, float | int | str] = {}
        stats.update(self._param_stats(self.tps_raw, self.tps_limit_px, "tps"))
        stats.update(self._param_stats(self.delaunay_raw, self.delaunay_limit_px, "delaunay"))
        stats.update(self._param_stats(self.roll_params, self.rolling_limit_px, "rolling"))
        stats.update(self._param_stats(self.dct_coeffs, self.dct_limit_px, "dct"))
        phase = self.fft_phase.raw_phase.detach().float()
        phase_limit = float(self.config.fft_phase_limit_rad)
        stats.update(
            {
                "fft_phase_num_at_min": int((phase <= -phase_limit + 1e-8).sum().cpu()),
                "fft_phase_num_at_max": int((phase >= phase_limit - 1e-8).sum().cpu()),
                "tps_enabled": int(self.config.tps_enabled),
                "delaunay_enabled": int(self.config.delaunay_enabled),
                "rolling_enabled": int(self.config.rolling_enabled),
                "dct_enabled": int(self.config.dct_enabled),
                "fft_phase_enabled": int(self.config.fft_phase_enabled),
            }
        )
        return stats

    def theta_state(self) -> dict[str, Any]:
        """Return only trainable perturbation parameters plus metadata.

        This intentionally excludes large fixed buffers such as TPS matrices,
        DCT bases, grids, and Delaunay interpolation weights. Those buffers are
        deterministic from config + image size and made previous `.pt` files
        unnecessarily huge.
        """

        return {
            "format": "wood_theta_only_v1",
            "height": self.height,
            "width": self.width,
            "channels": self.channels,
            "config": self.config.__dict__.copy(),
            "limits": self.limits_dict(),
            "theta": {
                "tps_raw": self.tps_raw.detach().cpu().clone(),
                "delaunay_raw": self.delaunay_raw.detach().cpu().clone(),
                "dct_coeffs": self.dct_coeffs.detach().cpu().clone(),
                "roll_params": self.roll_params.detach().cpu().clone(),
                "fft_phase_raw_phase": self.fft_phase.raw_phase.detach().cpu().clone(),
            },
        }

    def project_(self) -> dict[str, Any]:
        with torch.no_grad():
            blocks = [
                ("tps", self.tps_raw, self.tps_limit_px, self.config.tps_enabled),
                ("delaunay", self.delaunay_raw, self.delaunay_limit_px, self.config.delaunay_enabled),
                ("rolling", self.roll_params, self.rolling_limit_px, self.config.rolling_enabled),
                ("dct", self.dct_coeffs, self.dct_limit_px, self.config.dct_enabled),
            ]
            total_params = 0
            total_clamped = 0
            total_at_min = 0
            total_at_max = 0
            components = []
            for name, parameter, limit, enabled in blocks:
                parameter.nan_to_num_(0.0)
                if not enabled:
                    parameter.zero_()
                    continue
                before_low = parameter < -limit
                before_high = parameter > limit
                total_clamped += int((before_low | before_high).sum().item())
                parameter.clamp_(-limit, limit)
                at_min = int((parameter <= -limit + 1e-8).sum().item())
                at_max = int((parameter >= limit - 1e-8).sum().item())
                total_at_min += at_min
                total_at_max += at_max
                total_params += parameter.numel()
                if at_min or at_max:
                    components.append(name)
            if self.config.fft_phase_enabled:
                fft_stats = self.fft_phase.project_()
                phase = self.fft_phase.raw_phase
                total_params += phase.numel()
                total_clamped += int(fft_stats.get("fft_phase_num_clamped", 0))
                total_at_min += int(fft_stats.get("fft_phase_num_at_min", 0))
                total_at_max += int(fft_stats.get("fft_phase_num_at_max", 0))
                if fft_stats.get("fft_phase_num_at_min", 0) or fft_stats.get("fft_phase_num_at_max", 0):
                    components.append("fft_phase")
            else:
                self.fft_phase.raw_phase.zero_()
                fft_stats = {
                    "fft_phase_num_clamped": 0,
                    "fft_phase_num_at_min": 0,
                    "fft_phase_num_at_max": 0,
                }
            return {
                "num_total_params": int(total_params),
                "num_clamped_total": int(total_clamped),
                "fraction_clamped_total": float(total_clamped / max(total_params, 1)),
                "num_at_min_total": int(total_at_min),
                "num_at_max_total": int(total_at_max),
                "components_at_boundary": ",".join(sorted(set(components))),
                **fft_stats,
            }

    def limits_dict(self) -> dict[str, Any]:
        return {
            "tps_enabled": bool(self.config.tps_enabled),
            "delaunay_enabled": bool(self.config.delaunay_enabled),
            "rolling_enabled": bool(self.config.rolling_enabled),
            "dct_enabled": bool(self.config.dct_enabled),
            "fft_phase_enabled": bool(self.config.fft_phase_enabled),
            "tps_limit_px": self.tps_limit_px,
            "delaunay_limit_px": self.delaunay_limit_px,
            "rolling_limit_px": self.rolling_limit_px,
            "dct_limit_px": self.dct_limit_px,
            "tps_norm_limit": self.config.tps_norm_limit,
            "delaunay_norm_limit": self.config.delaunay_norm_limit,
            "rolling_norm_limit": self.config.rolling_norm_limit,
            "dct_norm_limit": self.config.dct_norm_limit,
            "fft_phase_limit_rad": float(self.config.fft_phase_limit_rad),
            "max_combined_disp_px": self.config.max_combined_disp_px,
        }
