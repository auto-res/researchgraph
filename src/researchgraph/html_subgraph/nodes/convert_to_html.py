import json
from jinja2 import Environment
from logging import getLogger
from pydantic import BaseModel

from researchgraph.utils.openai_client import openai_client
from researchgraph.writer_subgraph.nodes.paper_writing import WritingNode

logger = getLogger(__name__)


class LLMOutput(BaseModel):
    generated_html_text: str


def convert_to_html(
    llm_name: str,
    prompt_template: str,
    paper_content: dict[str, str],
) -> str | None:
    data = {
        "sections": [
            {"name": section, "content": paper_content[section]}
            for section in paper_content.keys()
        ]
    }

    env = Environment()
    template = env.from_string(prompt_template)
    prompt = template.render(data)

    messages = [
        {"role": "user", "content": prompt},
    ]
    response = openai_client(llm_name, message=messages, data_class=LLMOutput)
    if not response:
        logger.warning("LLM response is None.")
        return None
    response = json.loads(response)
    return response["generated_html_text"]

convert_to_html_prompt = """
You are an HTML expert. 
Your task is to convert each section of a research paper into **clean, semantic HTML**, suitable for publishing on a static site (such as GitHub Pages).

Below are the paper sections. For each one, generate HTML content. Use semantic HTML tags (e.g., <section>, <article>, <h2>, <p>, <ul>, <li>, <figure>, etc.)
{% for section in sections %}
---
Section: {{ section.name }}

{{ section.content }}

---
{% endfor %}

## HTML Formatting Rules:
- Use <section> with <h2> for each top-level section (e.g., Introduction, Method, Results, etc.).
    - The <h2> should match the section name.
    - Do not use nested sections unless truly needed.

- Use <p> tags for paragraphs.
    - Split logically separate thoughts into their own <p> blocks.

- For listing contributions or features, use <ul> and <li>.
    - Start each list item with <strong>...</strong> for a short descriptor.

- Use <figure> and <figcaption> with <img> for all figures (in the "Results" section only).
    - Use appropriate relative paths for src (e.g., images/foo.png).
    - Keep layout clear with centered or responsive figures.

- Use <pre><code> for pseudocode or actual code (only in the "Method" section).
    - Keep indentation clean and avoid syntax highlighting.

- Do not include any of the following:
    - <html>, <head>, <body>
    - Metadata, document titles, or <article> tags unless semantically essential

- All <a> tags must have target="_blank" and href attributes.
- Escape reserved characters like &, <, >.
- Avoid adding citations or references unless already included in the input.

- Do not invent or add any content (e.g., references, figures, lists) unless it exists in the input.

## Output Format:
```html
<!-- Title -->
<section>
  <h2>Title</h2>
  <p>...</p>
</section>

<!-- Abstract -->
<section>
  <h2>Abstract</h2>
  <p>...</p>
</section>
...
```
"""

if __name__ == "__main__":
    paper_content = {
        "Title": "Adaptive Multimodal Instruction and Co-Training (AMICT): A Compact Approach for On-Device Multimodal AI",
        "Abstract": "This paper introduces Adaptive Multimodal Instruction and Co-Training (AMICT), a lightweight transformer-based model that extends the compact design of BTLM-3B-8K to address key limitations in instruction-following, modality integration, and long-context understanding. AMICT enhances its predecessor’s capabilities through dual-stage multimodal pretraining, an advanced instruction-tuning phase reinforced with a lightweight reinforcement learning alignment loop, dynamic context modulation, and a hardware–software co-design strategy that enables efficient on-device inference. Trained on a diversified corpus of text, captioned images, and short audio snippets, the model is fine-tuned on interactive tasks such as chat and visual instruction following. We evaluate the approach using three Python-implementable experiments: a multimodal instruction-following evaluation, a long-context handling test with dynamic modulation, and an on-device inference benchmark. Experimental results from simulations using dummy models clearly indicate that AMICT achieves performance comparable to larger 7B-parameter systems, while significantly reducing memory and computational demands. These findings underscore the potential of AMICT for real-time, edge-based applications and pave the way for future work in instruction tuning, bias mitigation, and multilingual expansion.",
        "Introduction": "Recent advances in transformer-based language models have increasingly emphasized achieving robust context understanding and flexible task performance; however, these gains often come with heavy computational costs. The BTLM-3B-8K model demonstrated that a 3B-parameter system can occasionally match the performance of larger 7B models, particularly for tasks with long-context inputs. Nevertheless, the baseline model suffers from three main challenges: limited capability in following instructions, inability to incorporate non-textual modalities, and performance degradation when processing very long inputs. AMICT was conceived to overcome these shortcomings by integrating multimodal signals and adopting dynamic mechanisms for context handling while maintaining a compact and efficient architecture suitable for mobile and edge deployment.\n\nAMICT is founded on four core contributions. First, in the dual-stage multimodal pretraining phase, the model is trained using a combination of high-quality text, captioned image snippets, and short audio segments. Lightweight modular encoders process these additional modalities, and cross-modal attention blocks enable shared representations across different data types. Second, the enhanced instruction-tuning and alignment module employs interactive scenarios—ranging from conversational chat to visual instruction following—and a reinforcement learning loop that leverages external knowledge bases to improve factual accuracy and reduce bias. Third, dynamic context modulation replaces fixed context windows with adaptive attention mechanisms that adjust to the semantic density of input, thereby preserving performance even when the input exceeds 8K tokens. Finally, an integrated hardware–software co-design ensures that the model remains quantization-friendly, using approximately 3GB of memory at 4-bit precision, and achieves inference speeds comparable to 7B models while operating on modest computational resources.\n\nThe significance of these contributions lies in their potential to drive the development of practical, on-device multimodal AI systems capable of complex reasoning without the overhead of extremely large models. Our experimental setup simulates AMICT’s behavior and compares it to the baseline, with controlled tests on multimodal integration, long-context processing, and resource-efficient inference. The simulation results demonstrate improvements in contextual understanding and resource utilization, highlighting AMICT’s promise for real-world applications. Future work may focus on extending the approach to additional languages and modalities, refining training schedules, and integrating further bias mitigation strategies.",
        "Related Work": "Prior research on transformer-based language models has largely focused on scaling model parameters to boost performance. The BTLM-3B-8K model is a notable exception in which a 3B-parameter system achieves competitive performance relative to 7B models by employing a GPT-3–style architecture with modifications such as the SwiGLU activation, ALiBi positional embeddings, and a two-phase training process on a deduplicated SlimPajama corpus. While BTLM-3B-8K offers significant improvements in model compactness and inference efficiency, its exclusive focus on text limits its applicability in multimodal scenarios.\n\nConversely, studies inspired by Megrez-Omni have explored multimodal integration and edge-efficient training strategies, often separately addressing the incorporation of additional modalities or the challenges of long-context processing. These works underscore the value of utilizing image and audio data alongside text and highlight the importance of hardware-aware design for on-device deployment. AMICT distinguishes itself by unifying these advances into a single architecture, merging multimodal pretraining with dynamic attention mechanisms and efficient inference strategies. Our experimental comparisons further analyze performance metrics, computational overhead, and qualitative outputs in order to demonstrate the benefits of a unified multimodal approach.",
        "Background": "Understanding the innovations behind AMICT requires a review of several foundational concepts. The starting point is the GPT-3–style architecture, which is enhanced in BTLM-3B-8K by incorporating SwiGLU nonlinearity for improved training stability and ALiBi positional embeddings that enable better extrapolation over long input sequences. The original model was trained on the SlimPajama dataset using two distinct context lengths (2,048 and 8,192 tokens) and employed maximal update parameterization (µP) to transfer optimal hyperparameters from proxy models.\n\nA crucial advancement in AMICT is its dual-stage multimodal pretraining. At this stage, the model is exposed to a corpus that includes not only high-quality text, but also aligned image snippets—such as captioned images—and short segments of audio. Dedicated, lightweight encoders process non-textual data in parallel with the text module, and cross-modal attention blocks enable the fusion of features across modalities. Following this, the enhanced instruction-tuning phase further adapts the model by fine-tuning it on interactive scenarios. A lightweight reinforcement learning layer, inspired by recent alignment strategies, further refines the model by comparing outputs against dynamically updated knowledge sources.\n\nDynamic context modulation represents another key element. While the base model employs fixed context windows, AMICT introduces adaptive attention mechanisms that modulate based on the semantic density of the input, preserving effectiveness even when token counts exceed 8K. Finally, the hardware–software co-design ensures that the model is optimized for quantization and can run with low memory usage on edge devices. These design choices collectively ensure that the multimodal capabilities do not compromise speed or overall compactness.",
        "Method": "AMICT is constructed on the robust foundation of the BTLM-3B-8K backbone and extends it by integrating four interrelated modules:\n\n1. Dual-Stage Multimodal Pretraining\n   - The model is initially trained on a curated, deduplicated corpus that includes text, captioned images, and audio snippets. Dedicated yet lightweight encoders for image and audio data run parallel to the text encoder, merging outputs via cross-modal attention blocks. The underlying transformer architecture continues to utilize SwiGLU activations and ALiBi positional embeddings to ensure effective long-context processing.\n\n2. Enhanced Instruction-Tuning and Alignment\n   - In a further fine-tuning phase, the model is exposed to a variety of interactive tasks such as conversational chat, query-answering, and visual instruction following. A lightweight reinforcement learning loop, which employs web-based feedback and comparative ranking techniques, is integrated to reduce hallucinations and bias. This phase ensures that the model is better aligned with real-world application requirements.\n\n3. Dynamic Context Modulation\n   - Instead of relying on static context windows, AMICT introduces a dynamic modulation module that adjusts attention distributions based on the semantic density of inputs. This adaptive mechanism enables the model to maintain accuracy and coherence even when the input extends to 10K tokens or beyond.\n\n4. Hardware–Software Co-Design for On-Device Inference\n   - The final module focuses on maintaining efficiency. AMICT is optimized with quantization-friendly techniques, reducing memory usage to roughly 3GB at 4-bit precision, and achieving inference costs about 2.5× lower than typical 7B models. Modifications in resource scheduling between the multimodal submodules and the core transformer facilitate rapid and efficient on-device inference.\n\nThe careful integration of these components ensures that improvements in multimodal processing and long-context handling do not come at the expense of speed or efficiency.",
        "Experimental Setup": "The performance of AMICT is examined through three distinct experimental evaluations, each highlighting a specific advantage over the Base Method.\n\n1. Multimodal Instruction-Following Evaluation\n   - A benchmark dataset is constructed using paired text and image inputs, such as captioned images. In a PyTorch pipeline, both AMICT and the Base Method process the inputs. Images are preprocessed using standard transformations (resized to 224 × 224 pixels and normalized) and text is tokenized via the GPT-2 tokenizer from Hugging Face. Quantitative metrics (for example, BLEU scores) and qualitative assessments (such as visual inspection of output coherence) are recorded. A bar chart comparing response string lengths is produced as an initial dummy metric.\n\n2. Long-Context Handling and Dynamic Context Modulation Test\n   - This experiment involves processing long texts generated to have approximately 2K, 8K, and 10K tokens. Each input is fed to both models, and metrics such as generation latency, coherence, and response token counts are logged. Graphs plotting latency as a function of token length are generated to illustrate the effect of dynamic context modulation.\n\n3. On-Device Inference and Resource Efficiency Benchmark\n   - Dummy models simulating AMICT and the Base Method are deployed in a simulated on-device environment. Inference speed and memory footprint are measured using dynamic quantization and memory profiling tools. Using standard Python libraries (e.g., memory_profiler and torch’s benchmarking utilities), dual-axis bar charts are created to compare average inference latency and memory usage between the models.\n\nEach experiment is implemented via a complete Python script that covers all stages from data preprocessing to result visualization, ensuring reproducibility and clarity in the evaluation of the proposed improvements.",
        "Results": "The experimental evaluations yield several clear observations:\n\n- In the multimodal instruction-following experiment, AMICT effectively integrates image cues with text. Dummy outputs reflect that the model incorporates image statistics (for instance, mean values computed over input tensors) alongside textual metrics, resulting in richer and context-aware responses.\n\n- The long-context handling test shows that while both AMICT and the Base Method manage increased token counts, AMICT demonstrates a smoother performance degradation when processing inputs as long as 10K tokens. Latency measurements indicate minimal variation, with dynamic context modulation enabling sustained coherence and rapid response generation.\n\n- In the on-device inference benchmark, the hardware–software co-design presents near-zero differences in latency and memory usage compared to the baseline when measured on dummy models. Although both systems exhibit similar numerical outcomes in the simulated environment, these results highlight the potential for significant gains in real-world scenarios, where quantization and effective resource scheduling can lead to more notable improvements.\n\nFigures included in the results detail quantitative and qualitative comparisons. These comprise:\n- A bar chart comparing response string lengths from the multimodal evaluation.\n- Latency versus token length curves from the long-context experiment.\n- Dual-axis bar charts depicting inference latency and memory utilization in the simulated on-device setup.",
        "Conclusions": "In summary, Adaptive Multimodal Instruction and Co-Training (AMICT) offers a novel, compact solution for enhancing multimodal processing, dynamic long-context understanding, and efficient on-device inference. By building upon the BTLM-3B-8K architecture and integrating multiple innovations—including dual-stage multimodal pretraining, an enhanced instruction-tuning phase with reinforcement learning feedback, dynamic attention modulation, and a hardware–software co-design—AMICT successfully addresses the limitations of its predecessor. Our experiments, implemented via realistic Python-based evaluations, confirm that AMICT can achieve performance levels comparable to large 7B-parameter models while operating at a fraction of the computational and memory cost. Future research may focus on further refining dynamic context modulation, extending multimodal capabilities to additional languages and data modalities, and introducing explicit measures for bias mitigation. Overall, AMICT represents a significant step toward the deployment of efficient and versatile on-device AI systems."
    }
    llm_name = "o3-mini-2025-01-31"

    generated_html_text = convert_to_html(
        llm_name=llm_name,
        prompt_template=convert_to_html_prompt,
        paper_content=paper_content,
    )
    if not generated_html_text:
        print("Failed to generate html text.")
        exit(1)

    print("=== Generated HTML text ===")
    print(generated_html_text)
