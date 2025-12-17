from __future__ import annotations

import json
import traceback
from functools import lru_cache
from itertools import product
from os import path, walk
from typing import Generic, TypeVar, Union

import yaml


def to_dict(obj) -> dict:
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_dict(i) for i in obj]
    elif hasattr(obj, "__dict__"):
        result = {}
        for k, v in obj.__dict__.items():
            if v is not None:
                result[k] = to_dict(v)
        return result
    else:
        return obj


class TemplateBase:
    def __init__(self, file_path: str = ""):
        self.file_path = file_path

    def to_dict(self) -> dict:
        return to_dict(self)


TTemplate = TypeVar("TTemplate", bound=TemplateBase)
TTemplateOrDict = TypeVar("TTemplateOrDict", bound=Union[TemplateBase, dict])


class ListTemplate(TemplateBase):
    def __init__(self, id: str, file_path: str = ""):
        super().__init__(file_path)
        self.id = id


class ValueListTemplate(ListTemplate):
    def __init__(self, id, file_path="", values: list[str | int] = []):
        super().__init__(id, file_path)
        self.values = values


class Lists:
    def __init__(self, lists: dict[str, ListTemplate] = {}):
        self.lists = lists


class ExpansionRuleTemplate(TemplateBase):
    def __init__(self, id: str, file_path: str = "", rule: str = ""):
        super().__init__(file_path)
        self.id = id
        self.rule = rule


class ExpansionRules(TemplateBase):
    def __init__(self, file_path="", rules: dict[str, ExpansionRuleTemplate] = {}):
        super().__init__(file_path)
        self.rules = rules


class IntentTemplate(TemplateBase):
    def __init__(self, intent: str, file_path="", sentences: list[str] = []):
        super().__init__(file_path)
        self.intent = intent
        self.sentences = sentences


class Intents(TemplateBase):
    def __init__(self, file_path="", intents: dict[str, IntentTemplate] = {}):
        super().__init__(file_path)
        self.intents = intents


class Templates(TemplateBase):
    def __init__(
        self,
        file_path="",
        lists: dict[str, ListTemplate] = {},
        expansion_rules: dict[str, ExpansionRuleTemplate] = {},
        intents: dict[str, IntentTemplate] = {},
        skip_words: list[str] = [],
    ):
        super().__init__(file_path)
        self.lists = lists
        self.expansion_rules = expansion_rules
        self.intents = intents
        self.skip_words = skip_words


class ParseResult(Generic[TTemplateOrDict]):
    def __init__(
        self, file_path: str = "", errors: list[str] = [], templates: TTemplateOrDict = None
    ):
        self.file_path = file_path
        self.errors = errors
        self.templates = templates


TEMPLATE_DIR = path.join(path.dirname(__file__), "templates")


def load_yaml_file(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.full_load(file)


def parse_list_templates(file_path: str, data: dict) -> ParseResult[dict[str, ListTemplate]]:
    result = ParseResult[dict[str, ListTemplate]](file_path=file_path, templates={})
    for list_name, list_data in data.get("lists", {}).items():
        if "range" in list_data:
            try:
                result.templates[list_name] = ValueListTemplate(
                    id=list_name,
                    file_path=file_path,
                    values=[
                        str(n * int(list_data["range"].get("multiplyer", 1)))
                        for n in range(
                            int(list_data["range"]["from"]),
                            int(list_data["range"]["to"]) + 1,
                            round(
                                (
                                    int(list_data["range"]["to"])
                                    + 1
                                    - int(list_data["range"]["from"])
                                )
                                * 0.33
                            ),
                        )
                    ],
                )
            except Exception as e:
                result.errors.append(f"Error parsing range for {list_name} in {file_path}: {e}")
                traceback.print_exc()
        elif "values" in list_data:
            try:
                result.templates[list_name] = ValueListTemplate(
                    file_path=file_path,
                    id=list_name,
                    values=[
                        (
                            item
                            if isinstance(item, str)
                            else (item["out"] if "out" in item else item["in"])
                        )
                        for i, item in enumerate(list_data["values"])
                    ],
                )
            except Exception as e:
                result.errors.append(f"Error parsing values for {list_name} in {file_path}: {e}")
                traceback.print_exc()
        elif "wildcard" in list_data:
            pass
        else:
            result.errors.append(
                f"Unknown list type for {list_name} in {file_path}: {json.dumps(list_data)}"
            )
    return result


def parse_expansion_templates(
    file_path: str, data: dict
) -> ParseResult[dict[str, ExpansionRuleTemplate]]:
    result = ParseResult[dict[str, ExpansionRuleTemplate]](file_path=file_path, templates={})
    for rule_name, rule_data in data.get("expansion_rules", {}).items():
        result.templates[rule_name] = ExpansionRuleTemplate(
            file_path=file_path, id=rule_name, rule=rule_data
        )
    return result


def parse_intent_templates(
    file_path: str, data: dict
) -> (ParseResult[dict[str, IntentTemplate]], int):
    sentence_count = 0
    result = ParseResult[dict[str, IntentTemplate]](file_path=file_path, templates={})
    for intent_name, intent_data in data.get("intents", {}).items():
        sentences_list = []
        for item in intent_data.get("data", []):
            try:
                sentences_list.extend(item.get("sentences", []))
            except Exception as e:
                result.errors.append(
                    f"Error parsing sentences for intent {intent_name} in {file_path}: {e}"
                )
                traceback.print_exc()
        result.templates[intent_name] = IntentTemplate(
            file_path=file_path,
            intent=intent_name,
            sentences=sentences_list,
        )
        sentence_count += len(sentences_list)
    return result, sentence_count


def parse_yaml_file(file_path: str) -> ParseResult[Templates]:
    data = load_yaml_file(file_path)
    result = ParseResult[Templates](file_path=file_path, templates=Templates(file_path=file_path))
    lists_result = parse_list_templates(file_path, data)
    result.templates.lists = lists_result.templates

    expansion_rules_result = parse_expansion_templates(file_path, data)
    result.templates.expansion_rules = expansion_rules_result.templates

    intents_result, sentence_count = parse_intent_templates(file_path, data)
    result.templates.intents = intents_result.templates

    result.templates.errors = (
        lists_result.errors + expansion_rules_result.errors + intents_result.errors
    )
    return result, sentence_count


def load_all_templates(directory: str) -> (ParseResult[Templates], int):
    sentence_count = 0
    all_templates = Templates(file_path=directory)
    all_errors = []

    for root, _, files in walk(directory):
        for file in files:
            if file.endswith(".yaml") or file.endswith(".yml"):
                file_path = path.join(root, file)
                result, file_sentences = parse_yaml_file(file_path)
                sentence_count += file_sentences
                duplicate_list_keys = all_templates.lists.keys() & result.templates.lists.keys()
                duplicate_expansion_keys = (
                    all_templates.expansion_rules.keys() & result.templates.expansion_rules.keys()
                )
                if duplicate_list_keys:
                    all_errors.extend(
                        [
                            f"Duplicate list keys {key} found in {all_templates.lists[key].file_path} and {result.templates.lists[key].file_path}"
                            for key in duplicate_list_keys
                        ]
                    )
                if duplicate_expansion_keys:
                    all_errors.extend(
                        [
                            f"Duplicate expansion keys {key} found in {all_templates.expansion_rules[key].file_path} and {result.templates.expansion_rules[key].file_path}"
                            for key in duplicate_expansion_keys
                        ]
                    )
                all_templates.lists.update(
                    {
                        k: v
                        for k, v in result.templates.lists.items()
                        if k not in duplicate_list_keys
                    }
                )
                all_templates.expansion_rules.update(
                    {
                        k: v
                        for k, v in result.templates.expansion_rules.items()
                        if k not in duplicate_expansion_keys
                    }
                )
                for intent, intent_data in result.templates.intents.items():
                    if intent not in all_templates.intents:
                        all_templates.intents[intent] = IntentTemplate(
                            file_path="", intent=intent, sentences=[]
                        )
                    all_templates.intents[intent].sentences.extend(intent_data.sentences)

                all_errors.extend(result.errors)

    return (
        ParseResult[Templates](file_path=directory, errors=all_errors, templates=all_templates),
        sentence_count,
    )


SPECIAL_CHARS = set("[({<")


@lru_cache(maxsize=None)
def match_group_cached(s, i, open_ch, close_ch):
    depth = 0
    for j in range(i, len(s)):
        if s[j] == open_ch:
            depth += 1
        elif s[j] == close_ch:
            depth -= 1
            if depth == 0:
                return s[i + 1 : j], j + 1
    raise ValueError(f"Unclosed {open_ch} starting at {i} {s}")


@lru_cache(maxsize=None)
def split_top_level_cached(s, sep="|"):
    parts, depth, buf = [], 0, []
    pairs = {"[": "]", "(": ")", "<": ">", "{": "}"}
    closes = set(pairs.values())

    for ch in s:
        if ch in pairs:
            depth += 1
        elif ch in closes:
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(buf))
            buf.clear()
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return tuple(parts)


def _prepare_lists(lists: dict[str, list | str]) -> dict[str, tuple]:
    prepped = {}
    for k, v in lists.items():
        if isinstance(v, str) and ".." in v:
            a, b = v.split("..")
            prepped[k] = tuple(str(n) for n in range(int(a), int(b) + 1))
        elif isinstance(v, str):
            prepped[k] = (v,)
        else:
            prepped[k] = tuple(dict.fromkeys(map(str, v)))
    return prepped


def _combine(groups):
    for parts in product(*groups):
        yield "".join(parts)


@lru_cache(maxsize=None)
def _expand_cached(expr, rules_id, lists_id):
    rules = _RULES[rules_id]
    lists = _LISTS[lists_id]
    tokens, i = [], 0
    special = SPECIAL_CHARS

    while i < len(expr):
        ch = expr[i]

        if ch == "[":
            inner, i = match_group_cached(expr, i, "[", "]")
            tokens.append(("", *expand(inner, rules, lists)))

        elif ch == "(":
            inner, i = match_group_cached(expr, i, "(", ")")
            opts = []
            for opt in split_top_level_cached(inner):
                opts.extend(expand(opt, rules, lists))
            tokens.append(tuple(opts))

        elif ch == "<":
            name, i = match_group_cached(expr, i, "<", ">")
            if name not in rules:
                raise KeyError(f"Unknown rule <{name}>")
            tokens.append(tuple(expand(rules[name], rules, lists)))

        elif ch == "{":
            name, i = match_group_cached(expr, i, "{", "}")
            if name not in lists:
                raise KeyError(f"Unknown list {{{name}}}")
            tokens.append(lists[name])

        else:
            start = i
            while i < len(expr) and expr[i] not in special:
                i += 1
            tokens.append((expr[start:i],))

    return tuple(s for s in (p.strip() for p in _combine(tokens)) if s)


_RULES = {}
_LISTS = {}


def expand(expr: str, rules: dict[str, str], lists: dict[str, tuple]):
    rid = id(rules)
    lid = id(lists)
    _RULES[rid] = rules
    _LISTS[lid] = lists
    return _expand_cached(expr, rid, lid)


def generate_sentences(expr: str, rules: dict[str, str], lists: dict[str, list | str]):
    prepped = _prepare_lists(lists)
    return list(set(expand(expr, rules, prepped)))


def generate_sentences_parallel(
    expr: str,
    rules: dict[str, str],
    lists: dict[str, list | str],
    *,
    max_workers: int = 4,
    branch_threshold: int = 32,
):
    from concurrent.futures import ProcessPoolExecutor

    prepped = _prepare_lists(lists)
    branches = split_top_level_cached(expr)

    if len(branches) < branch_threshold:
        return generate_sentences(expr, rules, prepped)

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        parts = pool.map(lambda b: expand(b, rules, prepped), branches)

    out = set()
    for p in parts:
        out.update(p)
    return list(out)


def generate_sentences_streaming(
    expr: str,
    rules: dict[str, str],
    lists: dict[str, list | str],
    consumer,
    formatter,
    *,
    dedupe: bool = False,
):
    """Stream expansions to a consumer callable without holding them all in memory."""
    prepped = _prepare_lists(lists)
    seen = set() if dedupe else None

    def emit(val: str):
        val = val.strip()
        if not val:
            return
        if seen is not None:
            if val in seen:
                return
            seen.add(val)
        consumer(formatter(val))

    def stream_expr(expr_text: str, stack: tuple[str, ...], cont):
        def rec(idx: int, current: str):
            if idx >= len(expr_text):
                cont(current)
                return

            ch = expr_text[idx]
            if ch == "[":
                inner, nxt = match_group_cached(expr_text, idx, "[", "]")
                rec(nxt, current)  # skip optional
                stream_expr(inner, stack, lambda val: rec(nxt, current + val))
            elif ch == "(":
                inner, nxt = match_group_cached(expr_text, idx, "(", ")")
                for opt in split_top_level_cached(inner):
                    stream_expr(opt, stack, lambda val: rec(nxt, current + val))
            elif ch == "<":
                name, nxt = match_group_cached(expr_text, idx, "<", ">")
                if name in stack:
                    raise ValueError(f"Cyclic rule reference: {' -> '.join(stack + (name,))}")
                if name in rules:
                    stream_expr(rules[name], stack + (name,), lambda val: rec(nxt, current + val))
                elif name in prepped:
                    for val in prepped[name]:
                        rec(nxt, current + str(val))
                else:
                    raise KeyError(f"Unknown rule or list reference: <{name}>")
            elif ch == "{":
                name, nxt = match_group_cached(expr_text, idx, "{", "}")
                parsed_name = name if ":" not in name else name.split(":")[0]
                if parsed_name not in prepped:
                    raise KeyError(f"Unknown list reference: {{{parsed_name}}}")
                for val in prepped[parsed_name]:
                    rec(nxt, current + str(val))
            else:
                start = idx
                while idx < len(expr_text) and expr_text[idx] not in SPECIAL_CHARS:
                    idx += 1
                literal = expr_text[start:idx]
                rec(idx, current + literal)

        rec(0, "")

    stream_expr(expr, tuple(), emit)


def get_route(intent: str) -> str:
    if intent.startswith("Hass"):
        return "CommandControl"
    if intent in ["query-media", "queue-media", "play-media"]:
        return "MediaLibrary"
    if intent in ["request-movie", "request-tv-show", "search-media"]:
        return "FetchMedia"
    return "Unknown"


if __name__ == "__main__":
    result, sentence_count = load_all_templates(TEMPLATE_DIR)
    if result.errors:
        print("Errors found while loading templates:")
        for error in result.errors:
            print(f"- {error}")
    else:
        # print(json.dumps(to_dict(result.templates.intents["Query-Media"].data), indent=2))
        rules = {k: v["rule"] for k, v in to_dict(result.templates.expansion_rules).items()}
        lists = {k: v["values"] for k, v in to_dict(result.templates.lists).items()}
        print(json.dumps(lists, indent=2))
        print("Templates loaded successfully.")
        progress = 0
        with open(path.join(__file__, "training.jsonl"), "w", encoding="utf-8") as f:

            def consume(sentence: str):
                f.write(sentence + "\n")

            for intent, intent_data in result.templates.intents.items():

                def format(sentence):
                    return json.dumps(
                        {
                            "text": sentence,
                            "label": get_route(intent),
                        }
                    )

                for sentence in intent_data.sentences:
                    generate_sentences_streaming(
                        sentence,
                        rules=rules,
                        lists=lists,
                        consumer=consume,
                        formatter=format,
                    )
                    if progress % 5 == 0:
                        print(f"Processed {progress} out of {sentence_count} templates...\n")
                    progress += 1
