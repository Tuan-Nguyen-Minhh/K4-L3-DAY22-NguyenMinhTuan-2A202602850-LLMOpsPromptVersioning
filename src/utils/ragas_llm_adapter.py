"""
Bọc LLM để RAGAS đọc kết quả ổn định.

Bài toán:
    1. Model Mistral luôn bọc output trong ```json ... ```. RAGAS không bóc
       fence nên parse thất bại; prompt sửa chữa (FixOutputFormat) lại bị
       model "chép lại" schema thay vì điền giá trị → parse hỏng vĩnh viễn.
    2. RAGAS cộng dồn token usage từ `llm_output` (kiểu dict) bằng `+=`.
       Với một số provider, dict đó không hỗ trợ `+=` →
       TypeError: unsupported operand type(s) for +=: 'dict' and 'dict'.

Cách xử lý: vẫn truyền cùng prompt, chỉ làm sạch kết quả trả về
(bóc fence, bỏ llm_output/usage_metadata). Không sửa thư viện RAGAS.
"""
import re
from typing import Any, List, Optional

from langchain_core.callbacks.manager import Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import Generation, LLMResult

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


def strip_fence(text: str) -> str:
    """Bỏ markdown code fence, trả về text thuần."""
    return _FENCE_RE.sub("", text).strip()


class RagasSafeLLM(BaseLanguageModel):
    """
    Proxy quanh một BaseLanguageModel.

    RAGAS gọi `agenerate_prompt()` rồi đọc `generation.text` để validate,
    và cộng dồn `llm_output` của từng lần gọi. Proxy này can thiệp đúng
    hai chỗ đó, mọi thứ khác giữ nguyên.
    """

    inner: Any

    @property
    def _llm_type(self) -> str:
        return "ragas-safe"

    @property
    def _identifying_params(self) -> dict:
        return {"wrapped": type(self.inner).__name__}

    # ── 2 method abstract của BaseLanguageModel: chuyển tiếp cho LLM bên trong ──
    def generate_prompt(self, *args: Any, **kwargs: Any) -> Any:
        return self.inner.generate_prompt(*args, **kwargs)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return self.inner.invoke(*args, **kwargs)

    # ── đường đồng bộ: RAGAS có thể dùng _generate ──────────────────────
    def _generate(self, *args: Any, **kwargs: Any) -> LLMResult:
        return self._clean(self.inner._generate(*args, **kwargs))

    async def _agenerate(self, *args: Any, **kwargs: Any) -> LLMResult:
        return self._clean(await self.inner._agenerate(*args, **kwargs))

    def generate(self, *args: Any, **kwargs: Any) -> LLMResult:
        return self._clean(self.inner.generate(*args, **kwargs))

    async def agenerate(self, *args: Any, **kwargs: Any) -> LLMResult:
        return self._clean(await self.inner.agenerate(*args, **kwargs))

    async def agenerate_prompt(
        self,
        prompts: List[Any],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Callbacks] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """
        Đường RAGAS thực sự gọi: `await langchain_llm.agenerate_prompt(prompts=...)`
        (ragas/llms/base.py:304) — nên method này phải là async.

        Cố tình gọi TỪNG prompt một thay vì gom lô: `answer_relevancy` yêu cầu
        n=3, và khi LangChain gom nhiều prompt lại, langchain-mistralai chạy
        `_combine_llm_outputs()` với đoạn `overall_token_usage[k] += v`
        trong đó v là dict → TypeError. Gọi lẻ nép đúng chỗ đó.
        """
        generations: List[Generation] = []
        for single in prompts:
            one = await self.inner.agenerate_prompt(
                [single], stop=stop, callbacks=callbacks, **kwargs
            )
            for gens in one.generations or []:
                generations.extend(gens)
        # RAGAS đọc `result.generations` rồi lấy `[0]` của từng phần tử
        # (ragas/llms/base.py:311) → mỗi phần tử phải là một list con.
        return self._clean(LLMResult(generations=[[g] for g in generations]))

    # ── làm sạch ────────────────────────────────────────────────────────
    @staticmethod
    def _clean(result: LLMResult) -> LLMResult:
        for generations in result.generations or []:
            for gen in generations:
                if isinstance(gen.text, str):
                    gen.text = strip_fence(gen.text)
        result.llm_output = None
        return result


def wrap_for_ragas(llm: BaseLanguageModel) -> BaseLanguageModel:
    """Bọc LLM, trừ khi đã được bọc rồi."""
    if isinstance(llm, RagasSafeLLM):
        return llm
    return RagasSafeLLM(inner=llm)
