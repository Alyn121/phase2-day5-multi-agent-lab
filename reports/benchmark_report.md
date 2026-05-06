# Benchmark Report: Sự khác biệt giữa RAG truyền thống và GraphRAG

| Run Name | Latency | Tokens (In/Out) | Cerebras Cost | OpenAI Est. | Quality | Sources |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 4.53s | 59/209 | $0.0000 | $0.0001 | 5.8/10 | 0 |
| multi-agent | 11.44s | 10874/8425 | $0.0000 | $0.0067 | 8.6/10 | 5 |

## Analysis
- **Cost Analysis:** Multi-agent with Cerebras saved approximately $0.0067 compared to OpenAI GPT-4o-mini rates.
- **Quality Gain:** Multi-agent achieved a +2.9 quality improvement.

### Agent Breakdown: baseline
| Agent | Calls | Tokens | % of Total |
|---|---:|---:|---:|
| researcher | 1 | 268 | 100.0% |

### Agent Breakdown: multi-agent
| Agent | Calls | Tokens | % of Total |
|---|---:|---:|---:|
| supervisor | 7 | 2149 | 11.1% |
| researcher | 1 | 3281 | 17.0% |
| analyst | 1 | 1294 | 6.7% |
| writer | 5 | 12575 | 65.2% |

## Judge Feedback

### Baseline
**Judge Reasoning:** Đáp án có thể không chính xác về các kỹ thuật cụ thể được sử dụng trong GraphRAG. Không có bất kỳ nguồn nào được trích dẫn để hỗ trợ thông tin. Đáp án chỉ đề cập đến các khái niệm cơ bản và không có sự phân tích sâu sắc về sự khác biệt giữa RAG truyền thống và GraphRAG. Cấu trúc và tone của đáp án tương đối chuyên nghiệp. (Acc: 8.0, Cit: 2.0, Dep: 6.0, Pre: 7.0)

### Multi-agent
**Judge Reasoning:** Đáp án cung cấp một cái nhìn tổng quan chi tiết về sự khác biệt giữa RAG truyền thống và GraphRAG, bao gồm cả ưu điểm, hạn chế và quy trình của GraphRAG. Tuy nhiên, có một số điểm cần cải thiện về độ chính xác, vì một số thông tin không được kiểm chứng hoặc không rõ ràng. Đáp án cũng cung cấp một số nguồn tham khảo, nhưng có thể cần thêm nguồn để tăng cường độ tin cậy. (Acc: 8.5, Cit: 9.0, Dep: 9.0, Pre: 8.0)
