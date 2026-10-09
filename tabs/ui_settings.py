"""Local WebUI preferences, separate from Applio's existing configuration.

Pages opt in explicitly. Their excluded/native settings never enter the store
or the reset outputs. Disk writes occur only on UI events, never audio callbacks.
"""
import copy
import json
import logging
import math
import os
import tempfile
import threading
from functools import partial
from pathlib import Path

import gradio as gr

_LOCK = threading.RLock()
_LOG = logging.getLogger(__name__)
_NOTE = "本頁參數會自動儲存。恢復預設值只影響此功能記憶的欄位。"


class UISettingsError(Exception):
    pass


class UISettingsStore:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else Path(__file__).resolve().parents[1] / "ui_settings.json"

    def _read(self):
        if not self.path.exists():
            return {"version": 1, "pages": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if (not isinstance(data, dict) or type(data.get("version")) is not int
                    or data["version"] != 1 or not isinstance(data.get("pages"), dict)
                    or any(not isinstance(page, dict) for page in data["pages"].values())):
                raise ValueError("Unsupported settings format")
            return data
        except (OSError, ValueError) as error:
            raise UISettingsError("無法讀取記憶設定；原檔案已保留。") from error

    def read_page(self, page):
        with _LOCK:
            return copy.deepcopy(self._read()["pages"].get(page, {}))

    def update_page(self, page, values):
        with _LOCK:
            temporary = None
            try:
                data = self._read()
                data["pages"].setdefault(page, {}).update(values)
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tmp",
                                                 prefix=self.path.name + ".", dir=self.path.parent,
                                                 delete=False) as output:
                    temporary = output.name
                    json.dump(data, output, ensure_ascii=False, indent=2, allow_nan=False)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, self.path)
            except (OSError, ValueError, UISettingsError) as error:
                raise UISettingsError("儲存失敗；先前的記憶設定已保留。") from error
            finally:
                if temporary is not None and os.path.exists(temporary):
                    os.unlink(temporary)

    def set_value(self, page, key, value):
        self.update_page(page, {key: value})


def conversion_visibility(components, values):
    """Initial/reset visibility for the conversion controls shared by pages."""
    visible = {}
    effects = {
        "reverb": ("room_size", "damping", "wet_gain", "dry_gain", "width", "freeze_mode"),
        "pitch_shift": ("semitones",),
        "limiter": ("threshold", "release_time"),
        "gain": ("db",),
        "distortion": ("gain",),
        "chorus": ("rate", "depth", "center_delay", "feedback", "mix"),
        "bitcrush": ("bit_depth",),
        "clipping": ("threshold",),
        "compressor": ("threshold", "ratio", "attack", "release"),
        "delay": ("seconds", "feedback", "mix"),
    }
    for suffix in ("", "_batch"):
        for source, targets in {
            "autotune": ("autotune_strength",),
            "proposed_pitch": ("proposed_pitch_threshold",),
            "clean_audio": ("clean_strength",),
            "formant_shifting": ("formant_row", "formant_preset", "formant_qfrency", "formant_timbre"),
        }.items():
            if source + suffix in values:
                for target in targets:
                    visible[target + suffix] = bool(values[source + suffix])
        if "embedder_model" + suffix in values:
            visible["embedder_custom" + suffix] = values["embedder_model" + suffix] == "custom"
        if "post_process" + suffix in values:
            enabled = bool(values["post_process" + suffix])
            for effect, parameters in effects.items():
                visible[effect + suffix] = enabled
                for parameter in parameters:
                    visible[effect + "_" + parameter + suffix] = enabled and bool(values.get(effect + suffix))
    return {name: flag for name, flag in visible.items() if name in components}


class PageSettings:
    def __init__(self, page, components, *, store, exclude, text_fields, path_fields,
                 allowed_values, visibility, reset_guard, choice_providers, context_fields, guard_fields):
        self.page, self.store, self.components = page, store, components
        self.path_fields, self.allowed_values = set(path_fields), allowed_values
        self.visibility, self.reset_guard = visibility, reset_guard
        self.choice_providers, self.context_fields = choice_providers, list(context_fields)
        self.guard_fields = list(guard_fields)
        self.context_defaults = {name: components[name].value for name in context_fields}
        supported = (gr.Slider, gr.Number, gr.Checkbox, gr.Dropdown, gr.Radio)
        self.fields = {
            name: component for name, component in components.items()
            if name not in exclude and not name.startswith("terms_checkbox")
            and getattr(component, "interactive", None) is not False
            and (isinstance(component, supported)
                 or (name in text_fields and isinstance(component, gr.Textbox)))
        }
        self.defaults = {name: copy.deepcopy(component.value) for name, component in self.fields.items()}
        values = dict(self.defaults)
        try:
            saved = store.read_page(page)
            invalid = []
            for name, value in saved.items():
                if name in self.fields:
                    if name in choice_providers or self.valid(name, value):
                        values[name] = value
                    else:
                        invalid.append(name)
            self.note = _NOTE if not invalid else "部分記憶選項已不可用，請重新選擇。" + _NOTE
        except UISettingsError as error:
            self.note = str(error) + _NOTE
        for name, choices in self._choices(values).items():
            self.fields[name].choices = choices
            if values[name] is not None and values[name] not in [item[1] for item in choices]:
                values[name] = self.defaults[name] if self.defaults[name] in [item[1] for item in choices] else None
        for name, value in values.items():
            self.fields[name].value = copy.deepcopy(value)
        flags = self._visibility(values)
        for name, flag in flags.items():
            components[name].visible = flag
        self.outputs = list(self.fields.values()) + [components[name] for name in flags if name not in self.fields]
        self.output_names = list(self.fields) + [name for name in flags if name not in self.fields]

    def valid(self, name, value, values=None):
        component = self.fields[name]
        if isinstance(component, gr.Checkbox):
            return type(value) is bool
        if isinstance(component, (gr.Slider, gr.Number)):
            return (type(value) in (int, float) and math.isfinite(value)
                    and (component.minimum is None or value >= component.minimum)
                    and (component.maximum is None or value <= component.maximum))
        if value is None:
            return not isinstance(component, gr.Textbox)
        if isinstance(component, gr.Textbox):
            valid = isinstance(value, str)
        else:
            if name in self.choice_providers:
                choices = [item[1] for item in self._choices(values or self.defaults)[name]]
            else:
                choices = self.allowed_values.get(name, [item[1] for item in component.choices])
            selected = value if getattr(component, "multiselect", False) and isinstance(value, list) else [value]
            valid = all(type(item) in (str, int, float, bool) and
                        (item in choices or getattr(component, "allow_custom_value", False)) for item in selected)
        if valid and name in self.path_fields and value:
            valid = isinstance(value, str) and Path(value).exists()
        return valid

    def _choices(self, values):
        context = {**self.context_defaults, **values}
        return {
            name: [(str(item[0]), item[1]) if isinstance(item, (tuple, list)) else (str(item), item)
                   for item in provider(context)]
            for name, provider in self.choice_providers.items()
        }

    def _visibility(self, values):
        flags = conversion_visibility(self.components, values)
        if self.visibility is not None:
            flags.update(self.visibility(values))
        return {name: bool(flag) for name, flag in flags.items() if name in self.components}

    def save(self, name, value, *context_values):
        try:
            # Dynamic dropdowns must validate against their current source,
            # e.g. a native realtime model that is intentionally not persisted here.
            if name in self.choice_providers:
                current = {**self.defaults, **self.store.read_page(self.page),
                           **dict(zip(self.context_fields, context_values)), name: value}
            else:
                current = None
            if not self.valid(name, value, current):
                return "儲存失敗：此欄位的值無效。"
            self.store.set_value(self.page, name, value)
            return "設定已儲存。" + _NOTE
        except UISettingsError as error:
            _LOG.warning("UI settings: %s", error)
            return str(error)

    def reset(self, *context_values):
        guard_values = context_values[len(self.context_fields):]
        context_values = context_values[:len(self.context_fields)]
        if self.reset_guard is not None and not self.reset_guard(*guard_values):
            raise gr.Error("請先停止 Realtime，再恢復本頁預設值。")
        context = {**self.defaults, **dict(zip(self.context_fields, context_values))}
        # Managed sources themselves reset to defaults; excluded sources retain
        # the current session value passed by the reset event.
        context.update(self.defaults)
        choices = self._choices(context)
        defaults = copy.deepcopy(self.defaults)
        for name, options in choices.items():
            if defaults[name] is not None and defaults[name] not in [item[1] for item in options]:
                defaults[name] = None
        try:
            # Update only registered keys, preserving every other page/key.
            self.store.update_page(self.page, defaults)
        except UISettingsError as error:
            raise gr.Error(str(error)) from error
        flags = self._visibility(defaults)
        updates = []
        for name in self.output_names:
            change = {}
            if name in defaults:
                change["value"] = defaults[name]
            if name in choices:
                change["choices"] = choices[name]
            if name in flags:
                change["visible"] = flags[name]
            updates.append(gr.update(**change))
        return updates + ["已恢復本頁預設值。原有自動記憶設定保持不變。"]


def remember_page(page, components, *, store=None, exclude=(), text_fields=(),
                  path_fields=(), allowed_values=None, visibility=None, reset_guard=None,
                  choice_providers=None, context_fields=(), guard_fields=()):
    binding = PageSettings(page, components, store=store or UISettingsStore(), exclude=set(exclude),
                           text_fields=set(text_fields), path_fields=path_fields,
                           allowed_values=allowed_values or {}, visibility=visibility, reset_guard=reset_guard,
                           choice_providers=choice_providers or {}, context_fields=context_fields, guard_fields=guard_fields)
    with gr.Row():
        reset_button = gr.Button("恢復本頁預設值", elem_id=f"ui-reset-{page}")
    status = gr.Markdown(binding.note)
    for name, component in binding.fields.items():
        component.change(fn=partial(binding.save, name),
                         inputs=[component] + [components[key] for key in context_fields], outputs=[status],
                         show_progress="hidden", trigger_mode="always_last",
                         concurrency_id="ui-settings", concurrency_limit=1)
    reset_button.click(fn=binding.reset, inputs=[components[key] for key in (*context_fields, *guard_fields)], outputs=binding.outputs + [status],
                       show_progress="hidden", concurrency_id="ui-settings", concurrency_limit=1)
    return binding
