# Phân tích kết quả — V1 vs V2

## 0. Liên kết

| | |
|---|---|
| **Repository** | https://github.com/Tuan-Nguyen-Minhh/K4-L3-DAY22-NguyenMinhTuan-2A202602850-LLMOpsPromptVersioning |
| **LangSmith project** (`day22-lab`) | https://smith.langchain.com/o/a63d6670-ed09-4964-84de-11ee6a286c8c/projects/p/7a04282f-5476-4d21-a036-e3708a334a8e |

Trên LangSmith, lọc theo `Run Name` để xem từng nhóm trace:

| Run name | Checkpoint | Số trace |
|---|---|---|
| `rag-query` | Nhiệm vụ 1 | 50 |
| `ab-rag-query` | Nhiệm vụ 2 | 50 |
| `ragas evaluation` | Nhiệm vụ 3 | 4 (2 phiên bản × 2 lần chạy) |

Tài liệu này giải thích **vì sao hai prompt khác nhau cho ra điểm khác nhau**, dựa trên
kết quả thực đo được trong Checkpoint 3 (`03_ragas_report.json`).

## 1. Cấu hình thử nghiệm

| Thành phần | Giá trị |
|---|---|
| Bộ QA | 50 cặp `question` / `reference` (`qa_pairs.py`) |
| Retrieval | FAISS, `chunk_size=500`, `chunk_overlap=50` → 107 chunks, `k=3` |
| LLM sinh câu trả lời | `ministral-8b-latest` (temperature = 0) |
| LLM chấm điểm | `ministral-8b-latest` (temperature = 0) |
| Embeddings | `mistral-embed-2312` (1024 dims) |
| Metrics | `faithfulness`, `answer_relevancy`, `context_recall`, `context_precision` |

Cả hai phiên bản chạy **cùng model, cùng tham số, cùng dữ liệu** nên điểm số so sánh
được với nhau.

Hai prompt được chạy lần lượt trên **toàn bộ 50 câu** (không phải A/B routing như
Checkpoint 2), nên mỗi phiên bản có một tập điểm riêng đầy đủ.

## 2. Kết quả

| Metric | V1 — ngắn gọn | V2 — có cấu trúc | Thắng |
|---|---|---|---|
| `faithfulness` | **0.9135** | 0.8548 | V1 |
| `answer_relevancy` | **0.9133** | 0.9095 | V1 |
| `context_recall` | 0.9800 | 0.9800 | hòa |
| `context_precision` | 0.9683 | **0.9800** | V2 |

Cả hai phiên bản đều đạt ngưỡng `faithfulness ≥ 0.8`.

## 3. Vì sao khác nhau

### 3.1. V1 thắng `faithfulness` — câu ngắn giảm rủi ro bịa đặt

`faithfulness` không hỏi "câu trả lời có đúng không" mà hỏi theo từng bước:

1. Tách câu trả lời thành các **claim** nguyên tử
2. Với từng claim, hỏi claim đó có được **context đã retrieve hỗ trợ** không
3. `faithfulness` = tỷ lệ claim được hỗ trợ

Prompt V1 yêu cầu **2–4 câu**, V2 yêu cầu **3–5 câu có tổ chức**. Câu trả lời ngắn
sống ít claim nên xác suất mọi claim đều có căn cứ trong context cao hơn.

Nói cách khác: `faithfulness` phạt **chi phí của việc nói thêm**. Mỗi câu thêm là
một claim có thêm, và mỗi claim là một cơ hội để nói ra thứ không có trong tài liệu.
V1 trả lời ít → ít cơ hội sai → `0.9135`. V2 nói nhiều hơn → nhiều cơ hội hơn →
`0.8548`.

Chênh lệch `0.059` là đúng quy mô mong đợi khi độ dài câu trả lời tăng khoảng 1–2 câu.

### 3.2. V2 thắng `context_precision` — chỉ trả lời đúng phần context đã lấy

`context_precision` đo: trong các đoạn đã retrieve, phần **liên quan tới câu trả lời**
chiếm bao nhiêu phần trăm.

Prompt V2 yêu cầu *"đọc kỹ context, xác định các facts liên quan"* — nó bắt model
tập trung vào đúng nội dung đã lấy về thay vì kể thêm kiến thức chung. Ở đây tôi dùng
prompt chạy trong **cùng điều kiện** với Checkpoint 1/2 (đề yêu cầu copy nguyên văn
`SYSTEM_V1`/`SYSTEM_V2` giữa các checkpoint để kết quả so sánh được), nên V2 giữ được
lợi thế này: `0.9800` so với `0.9683` của V1.

### 3.3. `context_recall` bằng nhau — retrieval không phải điểm nghẽn

Cả hai đều `0.9800`. Điều này nói rõ: **vấn đề không nằm ở retrieval**.

Kiểm chứng độc lập trước khi chấm: với `k=3`, 3 đoạn retrieve có chứa đủ các từ khoá
quan trọng của đáp án chuẩn cho **10/10 câu hỏi** đầu tiên. Retrieval đã tốt sẵn, nên
mọi khác biệt đều đến từ prompt.

### 3.4. Kết luận vận hành

Đây là đánh đổi **"nói ít nhưng chắc"** với **"nói nhiều và có cấu trúc"**:

| Nhu cầu | Nên chọn | Vì sao |
|---|---|---|
| Trả lời tự động, cần độ tin cậy cao | **V1** | `faithfulness` cao hơn 0.059 |
| Hỗ trợ người dùng, cần bao phủ và dễ đọc | **V2** | Câu dài có tổ chức, `context_precision` cao hơn |
| Miễn nhiễm cả hai | Prompt hỗn hợp | Chỉ định độ dài *và* ràng buộc "chỉ nêu fact có trong context" |

Bài học: **prompt dài hơn không tự động tốt hơn.** Một prompt tốt không phải prompt
viết nhiều, mà là prompt khiến mọi câu trả lời đều truy vết được về nguồn.

## 4. Ghi chú về môi trường chạy

Hai điểm dưới đây là **giới hạn của provider miễn phí**, không phải yêu cầu của đề.
Đề bài gợi ý chạy với `gpt-4o-mini`; tôi dùng Mistral free tier vì tài khoản
OpenRouter đã hết credits.

### 4.1. `max_workers=8`

RAGAS mặc định `max_workers=16`. Trên OpenRouter, 16 worker song song vượt giới hạn
`in_flight_budget` và làm **mọi** job chấm điểm thất bại → toàn bộ điểm ra `NaN`.
Đã đặt `max_workers=8`, vẫn dưới ngưỡng 1 request/giây của Mistral.

### 4.2. `src/utils/ragas_llm_adapter.py`

Đây là lớp bọc LLM, **không sửa thư viện RAGAS**. Nó xử lý 2 vấn đề đã kiểm chứng
bằng traceback:

1. **Bug `langchain-mistralai`** — `chat_models.py:702` chạy
   `overall_token_usage[k] += v` trong khi `v` là `dict`, gây
   `TypeError: unsupported operand type(s) for +=: 'dict' and 'dict'`.
   Lỗi chỉ xảy ra khi gom lô nhiều prompt (`n>1`), mà `answer_relevancy` là metric
   duy nhất yêu cầu `n=3`. Adapter gọi từng prompt lẻ để nép đúng chỗ đó.

2. **Markdown fence** — model Mistral luôn bọc output trong ```json ... ```, RAGAS
   không bóc fence nên parse thất bại. Adapter bóc fence trước khi trả về.

Không có adapter này, `faithfulness` và `answer_relevancy` trả về `NaN` **bất kể
dùng model nào qua Mistral**. Kèm theo đó là đội hình người học sinh gặp phải khi
chạy bài này với provider miễn phí.

Phần còn lại của dự án (chunking, retrieval, dựng LCEL chain, định nghĩa 4 metric)
giữ nguyên như đề bài yêu cầu.
