using HTTP
using JSON3

# ==============================================================================
# CONFIGURATION: Replace with your actual token from hf.co/settings/tokens
# ==============================================================================
const HF_TOKEN = "your_hf_token_here" 

function call_deepseek_reasoning(prompt::String; token=HF_TOKEN)
    # The new mandatory Router endpoint
    url = "https://router.huggingface.co/hf-inference/models/deepseek-ai/DeepSeek-R1"
    
    if token == "your_hf_token_here" || isempty(token)
        error("Missing API Token. Please paste your Hugging Face token into the HF_TOKEN variable.")
    end

    headers = [
        "Authorization" => "Bearer $token",
        "Content-Type" => "application/json"
    ]
    
    # Request body structure for Chat Completion
    body = JSON3.write(Dict(
        "model" => "deepseek-ai/DeepSeek-R1",
        "messages" => [
            Dict("role" => "user", "content" => prompt)
        ],
        "max_tokens" => 1000, # Increased for deep reasoning
        "stream" => false
    ))

    println("Requesting reasoning from DeepSeek-R1...")

    try
        response = HTTP.post(url, headers, body)
        data = JSON3.read(response.body)
        
        # Extract the content from the response JSON
        return data[:choices][1][:message][:content]
    catch e
        if e isa HTTP.Exceptions.StatusError
            # Handles 401 (Unauthorized) or 403 (Gated Model) errors
            println("Error Code: ", e.status)
            println("Response: ", String(e.response.body))
        end
        rethrow(e)
    end
end

# ==============================================================================
# EXECUTION: Replicating the Syllogism Proof
# ==============================================================================
syllogism_prompt = """
Prove that syllogism:
All organic organisms are mortal. 
Humans are organic organisms. 
Thus humans are mortal. 

Provide as many self-reasoning thoughts you can as you provide a response.
"""

try
    response_text = call_deepseek_reasoning(syllogism_prompt)
    
    println("\n" * "="^30)
    println("DEEPSEEK REASONING OUTPUT")
    println("="^30 * "\n")
    
    # Julia prints the string directly; no need for Python's textwrap 
    # if using a modern terminal emulator.
    println(response_text)
catch e
    println("\nFailed to get response. Please ensure your token is valid and you have accepted the model terms at: https://huggingface.co/deepseek-ai/DeepSeek-R1")
end

