import os
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline


def load_image(image_path, mode="RGB"):
    """
    画像を読み込み、指定したモードに変換します。
    """
    return Image.open(image_path).convert(mode)


def main():
    # 環境変数からHugging Faceのトークンを取得
    huggingface_token = "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"
    if huggingface_token is None:
        raise ValueError("HUGGINGFACE_HUB_TOKEN 環境変数が設定されていません。")

    # ベース画像とマスク画像のパス
    base_image_path = "input/room.png"
    mask_image_path = "input/mask.png"
    output_image_path = "output/inpainted_image.png"

    # 画像の読み込み
    try:
        base_image = load_image(base_image_path, mode="RGB")
        mask_image = load_image(mask_image_path, mode="L")  # マスクはグレースケール
    except Exception as e:
        raise RuntimeError(f"画像の読み込みに失敗しました: {e}")

    # パイプラインの初期化
    model_id = "runwayml/stable-diffusion-inpainting"  # インペインティング専用モデル
    try:
        pipe: StableDiffusionInpaintPipeline = (
            StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_auth_token="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",
            ).to("cuda")
        )

        # テキストの反転埋め込みをロード
        try:
            pipe.load_textual_inversion(
                pretrained_model_name_or_path="sd-concepts-library/azura-from-vibrant-venture",
                token="embedding",
            )
            print("埋め込みが正常にロードされました。")
        except Exception as e:
            raise RuntimeError(f"埋め込みのロードに失敗しました: {e}")

    except Exception as e:
        raise RuntimeError(f"パイプラインの初期化に失敗しました: {e}")

    # プロンプトの設定
    prompt = "floor of embedding"
    negative_prompt = "low quality, blurry, unrealistic, less detailed"

    # 生成の設定
    seed = 10000
    generator = torch.Generator("cuda").manual_seed(seed)

    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            image=base_image,
            mask_image=mask_image,
            num_inference_steps=50,  # ステップ数の調整
            guidance_scale=15,  # ガイダンススケールの調整
        ).images[0]

        # 生成された画像を元の画像サイズにリサイズ
        result = result.resize(base_image.size)
    except Exception as e:
        raise RuntimeError(f"インペインティングの実行に失敗しました: {e}")

    # 生成された画像の保存
    try:
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        result.save(output_image_path)
        print(f"生成された画像を保存しました: {output_image_path}")
    except Exception as e:
        raise RuntimeError(f"画像の保存に失敗しました: {e}")


if __name__ == "__main__":
    main()
