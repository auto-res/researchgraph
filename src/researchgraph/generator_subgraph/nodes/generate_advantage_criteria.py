from researchgraph.utils.openai_client import openai_client


def generate_advantage_criteria(model_name: str, new_method: str) -> str:
    prompt = f"""
Please follow the instructions below and tell us about your experimental plan to demonstrate the superiority of the “New Method”.
- Please tell us up to three things you would like to experiment with.
- Please make sure that the things you would like to experiment with are realistic and possible to code in python.

# New Methods
----------------------------------------
{new_method}
----------------------------------------"""
    output = openai_client(
        model_name=model_name,
        prompt=prompt,
    )

    return output


if __name__ == "__main__":
    # model_name = "gpt-4.5-preview-2025-02-27"
    model_name = "o3-mini-2025-01-31"
    # model_name = "o1-2024-12-17"
    new_method = """
Adaptive Curvature Momentum (ACM) Optimizer Overview Existing adaptive optimizers such as Adam and AdaBelief dynamically adjust the learning rate based on the history of gradients. However, while these methods adapt to the magnitude of the gradients, they do not fully exploit information about the local curvature of the loss landscape. In this proposal, we introduce a new optimizer called Adaptive Curvature Momentum (ACM), which utilizes local quadratic approximations to adaptively adjust the update direction and scale. Method Standard Momentum Update Similar to SGD or Adam, ACM maintains a momentum term based on past gradients. Adaptive Learning Rate Scaling Uses second-order information (approximations of the Hessian) to dynamically adjust the learning rate for each direction. To reduce the computational cost of Hessian calculations, Fisher Information Matrix approximations can be employed. Curvature-Aware Adaptive Adjustment Estimates curvature by using the gradient change rate: Δ 𝑔 = 𝑔 𝑡 − 𝑔 𝑡 − 1 Δg=g t ​ −g t−1 ​ Modifies the learning rate based on curvature: 𝜂 𝑡 = 𝛼 1 + 𝛽 ⋅ Curvature ( 𝑔 𝑡 ) η t ​ = 1+β⋅Curvature(g t ​ ) α ​ where 𝛼 α is the base learning rate, and 𝛽 β controls the influence of curvature. Adaptive Regularization Encourages stable updates by incorporating an adaptive weight decay mechanism. When local curvature is high, the optimizer strengthens regularization to suppress excessive updates. Key Features and Benefits ✅ Combines Adam-style adaptability with curvature-aware updates ✅ Faster convergence: Adapts step sizes dynamically, taking larger steps in flat regions and smaller steps in sharp valleys. ✅ Hessian-free approximation: Utilizes efficient curvature estimation while maintaining low computational overhead. ✅ Scalability: Suitable for large-scale models such as ResNets and Transformers.
"""
    output = generate_advantage_criteria(
        model_name=model_name,
        new_method=new_method,
    )
    print(output)
