# WOOD: InstructPix2Pix White-box Geometry Results

Combined differentiable perturbation results

Author: Parth Katiyar

## Method

WOOD optimizes `Z` with `loss = -Z`. Model weights are frozen; only differentiable perturbation parameters are optimized.

## Run matrix

| model | code objective | objective | cases | iterations | status |
| --- | --- | --- | --- | --- | --- |
| InstructPix2Pix | vae_conditioning | VAE conditioning latent | 4.0000 | 300.00 | done |
| InstructPix2Pix | unet_prediction | UNet denoising prediction | 4.0000 | 300.00 | done |

## Aggregate summary

| model | objective | runs | mean final Z | mean SSIM original | mean output SSIM | mean output L2 |
| --- | --- | --- | --- | --- | --- | --- |
| InstructPix2Pix | VAE conditioning latent | 4.0000 | 129.06 | 0.9833 | 0.8309 | 0.0520 |
| InstructPix2Pix | UNet denoising prediction | 4.0000 | 0.0026 | 0.9952 | 0.8802 | 0.0358 |

## Per-run final values

| objective | face | prompt | final Z | final loss | SSIM original | output SSIM | output L2 | max disp px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VAE conditioning latent | face_002 | add black sunglasses | 127.66 | -127.66 | 0.9978 | 0.9105 | 0.0283 | 1.5777 |
| VAE conditioning latent | face_002 | add headphones | 127.55 | -127.55 | 0.9973 | 0.8973 | 0.0331 | 1.6837 |
| VAE conditioning latent | face_005 | add black sunglasses | 131.70 | -131.70 | 0.9493 | 0.6895 | 0.0965 | 26.253 |
| VAE conditioning latent | face_005 | add headphones | 129.32 | -129.32 | 0.9890 | 0.8263 | 0.0503 | 11.527 |
| UNet denoising prediction | face_002 | add black sunglasses | 0.0024 | -0.0024 | 0.9951 | 0.8674 | 0.0368 | 2.7150 |
| UNet denoising prediction | face_002 | add headphones | 0.0024 | -0.0024 | 0.9959 | 0.8633 | 0.0378 | 3.2953 |
| UNet denoising prediction | face_005 | add black sunglasses | 0.0029 | -0.0029 | 0.9955 | 0.9136 | 0.0293 | 3.0420 |
| UNet denoising prediction | face_005 | add headphones | 0.0029 | -0.0029 | 0.9942 | 0.8763 | 0.0392 | 3.3805 |

## Image strips

### VAE conditioning latent / face_002 / add black sunglasses

![strip](assets/strips/wood_vae_conditioning_face_002_add_black_sunglasses.png)

### VAE conditioning latent / face_002 / add headphones

![strip](assets/strips/wood_vae_conditioning_face_002_add_headphones.png)

### VAE conditioning latent / face_005 / add black sunglasses

![strip](assets/strips/wood_vae_conditioning_face_005_add_black_sunglasses.png)

### VAE conditioning latent / face_005 / add headphones

![strip](assets/strips/wood_vae_conditioning_face_005_add_headphones.png)

### UNet denoising prediction / face_002 / add black sunglasses

![strip](assets/strips/wood_unet_prediction_face_002_add_black_sunglasses.png)

### UNet denoising prediction / face_002 / add headphones

![strip](assets/strips/wood_unet_prediction_face_002_add_headphones.png)

### UNet denoising prediction / face_005 / add black sunglasses

![strip](assets/strips/wood_unet_prediction_face_005_add_black_sunglasses.png)

### UNet denoising prediction / face_005 / add headphones

![strip](assets/strips/wood_unet_prediction_face_005_add_headphones.png)

## Graphs

### VAE conditioning latent

#### VAE conditioning latent: Z vs iteration

![VAE conditioning latent: Z vs iteration](assets/graphs/vae_conditioning_latent_Z.png)

#### VAE conditioning latent: loss vs iteration

![VAE conditioning latent: loss vs iteration](assets/graphs/vae_conditioning_latent_loss.png)

#### SSIM and PSNR to original

![SSIM and PSNR to original](assets/graphs/vae_conditioning_latent_ssim_psnr.png)

#### Geometry component contribution

![Geometry component contribution](assets/graphs/vae_conditioning_latent_components_raw.png)

#### Geometry component contribution normalized

![Geometry component contribution normalized](assets/graphs/vae_conditioning_latent_components_normalized.png)

### UNet denoising prediction

#### UNet denoising prediction: Z vs iteration

![UNet denoising prediction: Z vs iteration](assets/graphs/unet_denoising_prediction_Z.png)

#### UNet denoising prediction: loss vs iteration

![UNet denoising prediction: loss vs iteration](assets/graphs/unet_denoising_prediction_loss.png)

#### SSIM and PSNR to original

![SSIM and PSNR to original](assets/graphs/unet_denoising_prediction_ssim_psnr.png)

#### Geometry component contribution

![Geometry component contribution](assets/graphs/unet_denoising_prediction_components_raw.png)

#### Geometry component contribution normalized

![Geometry component contribution normalized](assets/graphs/unet_denoising_prediction_components_normalized.png)
