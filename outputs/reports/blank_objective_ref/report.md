# WOOD blank-objective-reference report

WOOD optimizes the scalar `Z` with `loss = -Z` for InstructPix2Pix only.
The blank image is used as the objective-space reference, not as a visual counter-loss.

Completed runs found: 8

## Per-run image index

### unet_prediction / face_002 / add black sunglasses

- final Z: 0.0022692037746310234
- final loss: -0.0022692037746310234
- final SSIM to original: 1.0
- final output SSIM: 1.0
- run dir: `outputs/blank_objective_ref/20260630_062726_blank_objective_ref_all_sequential/runs/blank_objective_ref/instruct/unet_prediction/face_002__add_black_sunglasses`
- comparison sheet: `outputs/blank_objective_ref/20260630_062726_blank_objective_ref_all_sequential/runs/blank_objective_ref/instruct/unet_prediction/face_002__add_black_sunglasses/comparison_sheet.png`

### unet_prediction / face_002 / add headphones

- final Z: 0.0022707083262503147
- final loss: -0.0022707083262503147
- final SSIM to original: 1.0
- final output SSIM: 1.0
- run dir: `outputs/blank_objective_ref/20260630_062726_blank_objective_ref_all_sequential/runs/blank_objective_ref/instruct/unet_prediction/face_002__add_headphones`
- comparison sheet: `outputs/blank_objective_ref/20260630_062726_blank_objective_ref_all_sequential/runs/blank_objective_ref/instruct/unet_prediction/face_002__add_headphones/comparison_sheet.png`

### unet_prediction / face_005 / add black sunglasses

- final Z: 0.002735602203756571
- final loss: -0.002735602203756571
- final SSIM to original: 1.0
- final output SSIM: 1.0
- run dir: `outputs/blank_objective_ref/20260630_062726_blank_objective_ref_all_sequential/runs/blank_objective_ref/instruct/unet_prediction/face_005__add_black_sunglasses`
- comparison sheet: `outputs/blank_objective_ref/20260630_062726_blank_objective_ref_all_sequential/runs/blank_objective_ref/instruct/unet_prediction/face_005__add_black_sunglasses/comparison_sheet.png`

### unet_prediction / face_005 / add headphones

- final Z: 0.002739951480180025
- final loss: -0.002739951480180025
- final SSIM to original: 1.0
- final output SSIM: 1.0
- run dir: `outputs/blank_objective_ref/20260630_062726_blank_objective_ref_all_sequential/runs/blank_objective_ref/instruct/unet_prediction/face_005__add_headphones`
- comparison sheet: `outputs/blank_objective_ref/20260630_062726_blank_objective_ref_all_sequential/runs/blank_objective_ref/instruct/unet_prediction/face_005__add_headphones/comparison_sheet.png`

### vae_conditioning / face_002 / add black sunglasses

- final Z: 148.5875701904297
- final loss: -148.5875701904297
- final SSIM to original: 0.6398299336433411
- final output SSIM: 0.4604129493236542
- run dir: `outputs/blank_objective_ref/20260630_062726_blank_objective_ref_all_sequential/runs/blank_objective_ref/instruct/vae_conditioning/face_002__add_black_sunglasses`
- comparison sheet: `outputs/blank_objective_ref/20260630_062726_blank_objective_ref_all_sequential/runs/blank_objective_ref/instruct/vae_conditioning/face_002__add_black_sunglasses/comparison_sheet.png`

### vae_conditioning / face_002 / add headphones

- final Z: 148.61520385742188
- final loss: -148.61520385742188
- final SSIM to original: 0.6402467489242554
- final output SSIM: 0.44851163029670715
- run dir: `outputs/blank_objective_ref/20260630_062726_blank_objective_ref_all_sequential/runs/blank_objective_ref/instruct/vae_conditioning/face_002__add_headphones`
- comparison sheet: `outputs/blank_objective_ref/20260630_062726_blank_objective_ref_all_sequential/runs/blank_objective_ref/instruct/vae_conditioning/face_002__add_headphones/comparison_sheet.png`

### vae_conditioning / face_005 / add black sunglasses

- final Z: 151.65118408203125
- final loss: -151.65118408203125
- final SSIM to original: 0.5017560720443726
- final output SSIM: 0.65373295545578
- run dir: `outputs/blank_objective_ref/20260630_062726_blank_objective_ref_all_sequential/runs/blank_objective_ref/instruct/vae_conditioning/face_005__add_black_sunglasses`
- comparison sheet: `outputs/blank_objective_ref/20260630_062726_blank_objective_ref_all_sequential/runs/blank_objective_ref/instruct/vae_conditioning/face_005__add_black_sunglasses/comparison_sheet.png`

### vae_conditioning / face_005 / add headphones

- final Z: 151.72789001464844
- final loss: -151.72789001464844
- final SSIM to original: 0.5026940107345581
- final output SSIM: 0.48300662636756897
- run dir: `outputs/blank_objective_ref/20260630_062726_blank_objective_ref_all_sequential/runs/blank_objective_ref/instruct/vae_conditioning/face_005__add_headphones`
- comparison sheet: `outputs/blank_objective_ref/20260630_062726_blank_objective_ref_all_sequential/runs/blank_objective_ref/instruct/vae_conditioning/face_005__add_headphones/comparison_sheet.png`

## Graphs

![Z_curves](outputs/reports/blank_objective_ref/metric_curves/Z_curves.png)

![loss_curves](outputs/reports/blank_objective_ref/metric_curves/loss_curves.png)

![psnr_to_original_curves](outputs/reports/blank_objective_ref/metric_curves/psnr_to_original_curves.png)

![ssim_to_original_curves](outputs/reports/blank_objective_ref/metric_curves/ssim_to_original_curves.png)

![fraction_clamped_total_curves](outputs/reports/blank_objective_ref/metric_curves/fraction_clamped_total_curves.png)

![component_max_displacement_vs_iteration](outputs/reports/blank_objective_ref/metric_curves/component_max_displacement_vs_iteration.png)

![final_Z_vs_final_SSIM_scatter](outputs/reports/blank_objective_ref/metric_curves/final_Z_vs_final_SSIM_scatter.png)
