# OCI Generative AI Integration for Pythia

This module provides **OCI Generative AI** as an optional LLM backend for Pythia. It runs a local OpenAI-compatible proxy that authenticates with OCI using your `~/.oci/config` credentials and forwards requests to the OCI GenAI inference endpoint.

The default LLM backend remains **Ollama** -- this is an optional alternative for users who want to leverage OCI-hosted models. The default OCI model is **xai.grok-4**.

## Prerequisites

- Python 3.11+
- A configured `~/.oci/config` profile with valid OCI credentials
- An OCI compartment with access to the Generative AI service

## Quick Start

1. **Install dependencies:**
   ```bash
   cd oci-genai
   pip install -r requirements.txt
   ```

2. **Set environment variables:**
   ```bash
   export OCI_PROFILE=DEFAULT
   export OCI_REGION=us-chicago-1
   export OCI_COMPARTMENT_ID=ocid1.compartment.oc1..your-compartment-ocid
   export OCI_PROXY_API_KEY="$(python - <<'PY'
import secrets
print(secrets.token_urlsafe(32))
PY
)"
   ```

3. **Start the proxy:**
   ```bash
   python proxy.py
   # Proxy runs at http://localhost:9999/v1
   ```

4. **Configure Pythia** (`pythia.yaml`):
   ```yaml
   backend: oci-genai

   oci_genai:
     base_url: "http://localhost:9999/v1"
     model: "xai.grok-4"
     api_key: "same-value-as-OCI_PROXY_API_KEY"  # pragma: allowlist secret
     timeout_read: 180
   ```

5. **Or use the CLI flag ad hoc:**
   ```bash
   pythia query "What is RLHF?" --backend oci-genai
   pythia research "RISC-V vs ARM for edge AI" --backend oci-genai
   pythia serve --backend oci-genai
   ```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OCI_PROFILE` | `DEFAULT` | OCI config profile name from `~/.oci/config` |
| `OCI_REGION` | `us-chicago-1` | OCI region for the GenAI service endpoint |
| `OCI_COMPARTMENT_ID` | *(required)* | OCI compartment OCID |
| `OCI_PROXY_PORT` | `9999` | Local port for the proxy server |
| `OCI_PROXY_HOST` | `127.0.0.1` | Proxy bind host. Set a custom `OCI_PROXY_API_KEY` before binding to a non-loopback host. |
| `OCI_PROXY_API_KEY` | `oci-genai` | Bearer token required by the proxy. Override this for real use. |
| `OCI_PROXY_ALLOWED_ORIGIN` | `http://localhost:8900` | CORS origin for browser clients. |

## Available OCI GenAI Models

| Model ID | Description |
|---|---|
| `xai.grok-4` | **Default.** xAI Grok 4 (reasoning + non-reasoning) |
| `xai.grok-4.20` | xAI Grok 4.20 (reasoning + non-reasoning) |
| `xai.grok-4.20-multi-agent` | xAI Grok 4.20 Multi-Agent (real-time multi-agent research) |
| `meta.llama-3.3-70b-instruct` | Meta Llama 3.3 70B Instruct |
| `xai.grok-3-mini` | xAI Grok 3 Mini |
| `cohere.command-r-plus` | Cohere Command R+ |
| `cohere.command-r` | Cohere Command R |
| `meta.llama-3.1-405b-instruct` | Meta Llama 3.1 405B Instruct |

Model availability varies by region. Check the [OCI GenAI documentation](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm) for the latest list.

## How It Works

The proxy translates standard OpenAI API requests into OCI-authenticated requests:

```
Pythia (Python) --> localhost:9999/v1 (proxy.py) --> OCI GenAI endpoint
                    OpenAI-compatible              OCI User Principal Auth
```

- The proxy uses the `oci-openai` library which wraps the standard OpenAI Python client with OCI authentication
- Authentication uses OCI User Principal Auth from your `~/.oci/config` file
- The local proxy requires a bearer token before forwarding requests to OCI. Keep `oci_genai.api_key` in `pythia.yaml` aligned with `OCI_PROXY_API_KEY`.

## OCI Credentials Setup

If you don't have `~/.oci/config` set up yet:

```ini
[DEFAULT]
user=ocid1.user.oc1..aaaaaaaaexample
fingerprint=aa:bb:cc:dd:ee:ff:00:11:22:33:44:55:66:77:88:99
tenancy=ocid1.tenancy.oc1..aaaaaaaaexample
region=us-chicago-1
key_file=~/.oci/oci_api_key.pem
```

See the [OCI SDK Configuration](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdkconfig.htm) guide for details.

## Documentation

- [OCI Generative AI Service](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm)
- [oci-openai Python Library](https://pypi.org/project/oci-openai/)
- [OCI SDK Configuration](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdkconfig.htm)
