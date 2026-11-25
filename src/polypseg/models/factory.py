"""Сборка модели по конфигу.

Исходная модель - готовый U-Net из segmentation_models_pytorch (smp).
Модификации встраиваются в его декодер, энкодер и остальные блоки не меняются:
- model.aspp: true            -> центральный блок декодера заменяется на ASPP;
- model.attention_gates: true -> перед каждым блоком декодера skip-признаки
                                 проходят через Attention Gate.
"""

import segmentation_models_pytorch as smp
import torch.nn as nn

from polypseg.models.aspp import ASPP
from polypseg.models.attention_gate import AttentionGate


class AttentionGatedUnetDecoder(nn.Module):
    """Декодер U-Net из smp с Attention Gates на skip-соединениях.

    Блоки исходного декодера переиспользуются без изменений (ключи state_dict
    decoder.center.* и decoder.blocks.* те же, что у исходной модели). Изменён
    только способ объединения признаков: skip-признаки перед конкатенацией
    взвешиваются картой внимания.
    """

    def __init__(self, base_decoder, encoder_channels, decoder_channels):
        super().__init__()
        self.center = base_decoder.center
        self.blocks = base_decoder.blocks

        # Каналы считаются так же, как в smp.decoders.unet.decoder.UnetDecoder
        enc = list(encoder_channels[1:])[::-1]
        in_channels = [enc[0]] + list(decoder_channels[:-1])
        skip_channels = enc[1:] + [0]
        self.attention_gates = nn.ModuleList(
            AttentionGate(skip_ch, gate_ch) if skip_ch > 0 else nn.Identity()
            for skip_ch, gate_ch in zip(skip_channels, in_channels)
        )

    def forward(self, features):
        # Повторяет UnetDecoder.forward из smp 0.5.0 + вызов Attention Gate
        spatial_shapes = [f.shape[2:] for f in features][::-1]
        features = features[1:][::-1]
        head, skips = features[0], features[1:]

        x = self.center(head)
        for i, block in enumerate(self.blocks):
            height, width = spatial_shapes[i + 1]
            skip = skips[i] if i < len(skips) else None
            if skip is not None:
                # Сигнал g - выход предыдущего (более грубого) уровня декодера
                skip = self.attention_gates[i](skip, x)
            x = block(x, height, width, skip_connection=skip)
        return x


def build_model(model_cfg, pretrained=True):
    """Создаёт модель по секции model конфига.

    pretrained=False используется в сервисе и тестах: веса энкодера не скачиваются,
    потому что всё равно загружаются из чекпоинта.
    """
    encoder_weights = model_cfg.get("encoder_weights") if pretrained else None
    decoder_channels = tuple(model_cfg.get("decoder_channels", (256, 128, 64, 32, 16)))
    model = smp.Unet(
        encoder_name=model_cfg["encoder_name"],
        encoder_weights=encoder_weights,
        decoder_channels=decoder_channels,
        in_channels=3,
        classes=1,
        activation=None,
    )
    encoder_channels = model.encoder.out_channels

    if model_cfg.get("aspp", False):
        head_channels = encoder_channels[-1]
        model.decoder.center = ASPP(
            head_channels,
            head_channels,
            rates=tuple(model_cfg.get("aspp_rates", (1, 2, 4, 6))),
            branch_channels=model_cfg.get("aspp_branch_channels", 128),
            dropout=model_cfg.get("aspp_dropout", 0.1),
        )

    if model_cfg.get("attention_gates", False):
        model.decoder = AttentionGatedUnetDecoder(model.decoder, encoder_channels, decoder_channels)

    return model


def count_parameters(model):
    """Число обучаемых параметров (для сравнения вариантов в отчёте)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
