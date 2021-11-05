
from wheezy.captcha.image import (
    background,
    captcha,
    curve,
    noise,
    offset,
    rotate,
    smooth,
    text,
    warp,
)

if __name__ == "__main__":
    import random
    import string

    captcha_image = captcha(
        drawings=[
            background(),
            text(
                fonts=[
                    "fonts/CourierNew-Bold.ttf",
                    "fonts/LiberationMono-Bold.ttf",
                ],
                drawings=[warp(), rotate(), offset()],
            ),
            curve(),
            noise(),
            smooth(),
        ]
    )
    chars = string.digits + string.ascii_lowercase + string.ascii_uppercase
    image = captcha_image(random.sample(chars, 4))
    image.save("sample.png", "png]", quality=75)