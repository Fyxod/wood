# WOOD: InstructPix2Pix White-box Geometry Results

Combined differentiable perturbation results

Author: Parth Katiyar

## Method

WOOD optimizes `Z` with `loss = -Z`. Model weights are frozen; only differentiable perturbation parameters are optimized.

## Run matrix

| model | code objective | objective | cases | iterations | status |
| --- | --- | --- | --- | --- | --- |
| InstructPix2Pix | vae_conditioning | VAE conditioning latent | 4.0000 | 150.00 | done |
| InstructPix2Pix | unet_prediction | UNet denoising prediction | 4.0000 | 150.00 | done |

## Aggregate summary

| model | objective | runs | mean final Z | mean SSIM original | mean output SSIM | mean output L2 |
| --- | --- | --- | --- | --- | --- | --- |
| InstructPix2Pix | VAE conditioning latent | 4.0000 | 128.16 | 0.9966 | 0.9037 | 0.0322 |
| InstructPix2Pix | UNet denoising prediction | 4.0000 | 0.0026 | 0.9949 | 0.8745 | 0.0359 |

## Per-run final values

| objective | face | prompt | final Z | final loss | SSIM original | output SSIM | output L2 | max disp px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VAE conditioning latent | face_002 | add black sunglasses | 127.54 | -127.54 | 0.9961 | 0.8895 | 0.0319 | 2.5790 |
| VAE conditioning latent | face_002 | add headphones | 127.61 | -127.61 | 0.9980 | 0.9045 | 0.0312 | 1.3305 |
| VAE conditioning latent | face_005 | add black sunglasses | 128.73 | -128.73 | 0.9966 | 0.9275 | 0.0275 | 2.6911 |
| VAE conditioning latent | face_005 | add headphones | 128.77 | -128.77 | 0.9956 | 0.8932 | 0.0382 | 3.1632 |
| UNet denoising prediction | face_002 | add black sunglasses | 0.0024 | -0.0024 | 0.9946 | 0.8644 | 0.0364 | 3.9605 |
| UNet denoising prediction | face_002 | add headphones | 0.0024 | -0.0024 | 0.9949 | 0.8500 | 0.0412 | 3.0583 |
| UNet denoising prediction | face_005 | add black sunglasses | 0.0029 | -0.0029 | 0.9951 | 0.9021 | 0.0310 | 4.3791 |
| UNet denoising prediction | face_005 | add headphones | 0.0029 | -0.0029 | 0.9950 | 0.8814 | 0.0350 | 4.8791 |

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
