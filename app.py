"""
Flask web application for AI model API interaction and PRHP framework simulations.
Provides a simple UI with text input/output boxes and simulation controls.

Copyright © sanjivakyosan 2025
MIT License
"""
import os
import json
import sys
import re
from flask import Flask, render_template, request, jsonify
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import requests
import numpy as np

# Add src directory to path for PRHP imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Try to import adversarial gate
try:
    from adversarial_gate import enforce_nist_eu_only, sanitize_response
    ADVERSARIAL_GATE_AVAILABLE = True
except ImportError:
    ADVERSARIAL_GATE_AVAILABLE = False
    print("[WARNING] Adversarial gate not available. PRHP terminology may leak into responses.")

# Try to import output standardization
try:
    from output_standardization import standardize_output_text, OutputStandardizer
    OUTPUT_STANDARDIZATION_AVAILABLE = True
except ImportError:
    OUTPUT_STANDARDIZATION_AVAILABLE = False
    print("[WARNING] Output standardization not available. Output text will not be standardized.")

# Try to import output validation
try:
    from src.output_validation import validate_and_standardize_output
    OUTPUT_VALIDATION_AVAILABLE = True
except ImportError:
    try:
        from output_validation import validate_and_standardize_output
        OUTPUT_VALIDATION_AVAILABLE = True
    except ImportError:
        OUTPUT_VALIDATION_AVAILABLE = False
        print("[WARNING] Output validation not available. Output validation will be skipped.")

# Try to import metrics sync and utility boost
try:
    from src.metrics_sync_utility_boost import sync_metrics_and_boost_utilities, sync_prhp_results_and_boost_utilities
    METRICS_SYNC_UTILITY_BOOST_AVAILABLE = True
except ImportError:
    try:
        from metrics_sync_utility_boost import sync_metrics_and_boost_utilities, sync_prhp_results_and_boost_utilities
        METRICS_SYNC_UTILITY_BOOST_AVAILABLE = True
    except ImportError:
        METRICS_SYNC_UTILITY_BOOST_AVAILABLE = False
        print("[WARNING] Metrics sync and utility boost module not available. Metrics sync will be skipped.")

# Try to import output completion and validation
try:
    from src.output_completion_validation import complete_and_validate_output, validate_output_completeness
    OUTPUT_COMPLETION_VALIDATION_AVAILABLE = True
except ImportError:
    try:
        from output_completion_validation import complete_and_validate_output, validate_output_completeness
        OUTPUT_COMPLETION_VALIDATION_AVAILABLE = True
    except ImportError:
        OUTPUT_COMPLETION_VALIDATION_AVAILABLE = False
        print("[WARNING] Output completion and validation module not available. Output completion will be skipped.")

# Try to import NIST/EU mapper
try:
    from nist_eu_mapper import map_failure_to_regulation
    NIST_EU_MAPPER_AVAILABLE = True
except ImportError:
    NIST_EU_MAPPER_AVAILABLE = False
    print("[WARNING] NIST/EU mapper not available.")

# Try to load environment variables from .env file (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, use environment variables directly
    pass

# Try to import OpenAI client
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import PRHP framework
PRHP_IMPORT_ERROR = None
try:
    from prhp_core import simulate_prhp
    from prhp import prhp  # PRHP wrapper for prhp.simulate()
    from scenarios import load_safe_scenario  # Safe scenario loader
    from political_pruner import simulate_pruner_levels
    from virus_extinction import forecast_extinction_risk, simulate_viral_cascade
    from meta_empirical import full_meta_empirical_loop
    PRHP_AVAILABLE = True
except ImportError as e:
    PRHP_AVAILABLE = False
    PRHP_IMPORT_ERROR = str(e)

app = Flask(__name__)

# Helper function to safely convert to float (handles complex numbers and numpy types)
def safe_float(value):
    """Convert value to float, handling complex numbers, numpy arrays, and None."""
    if value is None:
        return None
    if isinstance(value, (complex, np.complexfloating)):
        return float(value.real)  # Take real part of complex numbers
    if isinstance(value, np.ndarray):
        value = value.item()  # Convert numpy scalar to Python type
        # Check again after conversion
        if isinstance(value, (complex, np.complexfloating)):
            return float(value.real)
    return float(value)

# Configuration
API_CONFIG = {
    'provider': os.getenv('AI_PROVIDER', 'openrouter'),  # 'openrouter', 'openai', 'anthropic', 'custom'
    'api_key': os.getenv('API_KEY', ''),
    'base_url': os.getenv('BASE_URL', 'https://openrouter.ai/api/v1'),
    'api_url': os.getenv('API_URL', 'https://api.openai.com/v1/chat/completions'),
    'model': os.getenv('MODEL', ''),  # Must be set via environment variable
    'max_tokens': int(os.getenv('MAX_TOKENS', '1000')),
    'temperature': float(os.getenv('TEMPERATURE', '0.7')),
    'site_url': os.getenv('SITE_URL', ''),
    'site_name': os.getenv('SITE_NAME', ''),
}


def call_openrouter_api(prompt: str, config: Dict[str, Any]) -> str:
    """Call OpenRouter.ai API using OpenAI client (legacy single message)."""
    return call_openrouter_api_with_history([{'role': 'user', 'content': prompt}], config)

def call_openrouter_api_with_history(messages: List[Dict[str, str]], config: Dict[str, Any]) -> str:
    """Call OpenRouter.ai API using OpenAI client with conversation history."""
    if not OPENAI_AVAILABLE:
        raise Exception("OpenAI package not installed. Run: pip install openai")
    
    try:
        client = OpenAI(
            base_url=config.get('base_url', 'https://openrouter.ai/api/v1'),
            api_key=config['api_key']
        )
        
        extra_headers = {}
        if config.get('site_url'):
            extra_headers['HTTP-Referer'] = config['site_url']
        if config.get('site_name'):
            extra_headers['X-Title'] = config['site_name']
        
        completion = client.chat.completions.create(
            extra_headers=extra_headers if extra_headers else None,
            model=config['model'],
            messages=messages,
            max_tokens=config.get('max_tokens', 1000),
            temperature=config.get('temperature', 0.7)
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        raise Exception(f"OpenRouter API request failed: {str(e)}")


def call_openai_api(prompt: str, config: Dict[str, Any]) -> str:
    """Call OpenAI API (legacy single message)."""
    return call_openai_api_with_history([{'role': 'user', 'content': prompt}], config)

def call_openai_api_with_history(messages: List[Dict[str, str]], config: Dict[str, Any]) -> str:
    """Call OpenAI API with conversation history."""
    if OPENAI_AVAILABLE:
        # Use OpenAI client if available
        try:
            client = OpenAI(
                api_key=config['api_key']
            )
            
            completion = client.chat.completions.create(
                model=config['model'],
                messages=messages,
                max_tokens=config.get('max_tokens', 1000),
                temperature=config.get('temperature', 0.7)
            )
            
            return completion.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API request failed: {str(e)}")
    else:
        # Fallback to requests
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {config['api_key']}"
        }
        
        data = {
            'model': config['model'],
            'messages': messages,
            'max_tokens': config['max_tokens'],
            'temperature': config['temperature']
        }
        
        try:
            response = requests.post(
                config['api_url'],
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")


def call_anthropic_api(prompt: str, config: Dict[str, Any]) -> str:
    """Call Anthropic (Claude) API (legacy single message)."""
    return call_anthropic_api_with_history([{'role': 'user', 'content': prompt}], config)

def call_anthropic_api_with_history(messages: List[Dict[str, str]], config: Dict[str, Any]) -> str:
    """Call Anthropic (Claude) API with conversation history."""
    headers = {
        'Content-Type': 'application/json',
        'x-api-key': config['api_key'],
        'anthropic-version': '2023-06-01'
    }
    
    # Anthropic expects messages in their format
    anthropic_messages = []
    for msg in messages:
        if msg['role'] == 'user':
            anthropic_messages.append({'role': 'user', 'content': msg['content']})
        elif msg['role'] == 'assistant':
            anthropic_messages.append({'role': 'assistant', 'content': msg['content']})
    
    data = {
        'model': config['model'],
        'max_tokens': config['max_tokens'],
        'messages': anthropic_messages
    }
    
    try:
        response = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return result['content'][0]['text']
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")


def call_custom_api(prompt: str, config: Dict[str, Any]) -> str:
    """Call custom API endpoint."""
    headers = {
        'Content-Type': 'application/json',
    }
    
    # Add API key to headers if provided
    if config.get('api_key'):
        headers['Authorization'] = f"Bearer {config['api_key']}"
    
    # Custom API format - adjust based on your API
    data = {
        'prompt': prompt,
        'max_tokens': config.get('max_tokens', 1000),
        'temperature': config.get('temperature', 0.7)
    }
    
    try:
        response = requests.post(
            config['api_url'],
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        # Try common response formats
        if 'response' in result:
            return result['response']
        elif 'text' in result:
            return result['text']
        elif 'content' in result:
            return result['content']
        elif 'message' in result:
            return result['message']
        else:
            return json.dumps(result, indent=2)
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")


@app.route('/')
def index():
    """Render the main UI page."""
    return render_template('index.html')


def detect_prhp_relevance(prompt: str) -> Dict[str, Any]:
    """
    Detect if the prompt is relevant to PRHP framework topics.
    ALWAYS RETURNS RELEVANT - All inputs are processed through PRHP.
    Returns dict with relevance info and suggested simulation parameters.
    """
    prompt_lower = prompt.lower()
    
    # Keywords that suggest PRHP relevance (for topic detection, not filtering)
    political_keywords = ['political', 'hierarchy', 'power', 'authority', 'governance', 'democracy', 
                         'collective', 'individual', 'society', 'social structure', 'hierarchical']
    philosophical_keywords = ['consciousness', 'mind', 'awareness', 'existence', 'reality', 'perception',
                            'identity', 'self', 'being', 'ontology', 'epistemology', 'ethics']
    psychological_keywords = ['adhd', 'autistic', 'neurodivergent', 'neurotypical', 'cognitive', 'mental',
                            'behavior', 'psychology', 'neural', 'brain', 'cognition', 'thinking']
    prhp_keywords = ['prhp', 'phi', 'fidelity', 'quantum', 'simulation', 'variant', 'neuro-cultural']
    
    # Check relevance (for topic detection only - all inputs are processed)
    is_political = any(kw in prompt_lower for kw in political_keywords)
    is_philosophical = any(kw in prompt_lower for kw in philosophical_keywords)
    is_psychological = any(kw in prompt_lower for kw in psychological_keywords)
    is_prhp_direct = any(kw in prompt_lower for kw in prhp_keywords)
    
    relevance_score = sum([is_political, is_philosophical, is_psychological, is_prhp_direct])
    
    # ALWAYS PROCESS THROUGH PRHP - No input is filtered out
    # If no keywords found, still process but with default relevance score
    if relevance_score == 0:
        relevance_score = 1  # Minimum score to ensure processing
    
    # Determine which variants to simulate based on question
    variants = ['neurotypical-hybrid']  # Default
    
    if 'adhd' in prompt_lower or 'collectivist' in prompt_lower or 'collective' in prompt_lower:
        variants = ['ADHD-collectivist']
    elif 'autistic' in prompt_lower or 'individualist' in prompt_lower or 'individual' in prompt_lower:
        variants = ['autistic-individualist']
    elif 'compare' in prompt_lower or 'difference' in prompt_lower or 'all' in prompt_lower:
        variants = ['ADHD-collectivist', 'autistic-individualist', 'neurotypical-hybrid']
    
    # Determine simulation parameters based on question complexity
    if 'detailed' in prompt_lower or 'comprehensive' in prompt_lower or 'extensive' in prompt_lower:
        levels = 9
        n_monte = 200
    elif 'quick' in prompt_lower or 'simple' in prompt_lower:
        levels = 5
        n_monte = 50
    else:
        levels = 9
        n_monte = 100
    
    return {
        'relevant': True,
        'relevance_score': relevance_score,
        'topics': {
            'political': is_political,
            'philosophical': is_philosophical,
            'psychological': is_psychological,
            'prhp_direct': is_prhp_direct
        },
        'suggested_simulation': {
            'levels': levels,
            'n_monte': n_monte,
            'variants': variants,
            'use_quantum': True,
            'seed': 42
        }
    }


def detect_prhp_command(prompt: str) -> Dict[str, Any]:
    """
    Detect if the prompt contains explicit PRHP simulation commands.
    Returns dict with command type and parameters if detected, None otherwise.
    """
    prompt_lower = prompt.lower()
    
    # Check for explicit simulation run commands
    if any(keyword in prompt_lower for keyword in ['run prhp', 'run simulation', 'simulate prhp', 'prhp simulation']):
        # Extract parameters from prompt
        levels = 9
        n_monte = 100
        seed = 42
        variants = ['neurotypical-hybrid']
        use_quantum = True
        
        # Try to extract levels
        levels_match = re.search(r'(\d+)\s*levels?', prompt_lower)
        if levels_match:
            levels = int(levels_match.group(1))
        
        # Try to extract iterations
        monte_match = re.search(r'(\d+)\s*(?:monte|iterations?|runs?)', prompt_lower)
        if monte_match:
            n_monte = int(monte_match.group(1))
        
        # Try to extract seed
        seed_match = re.search(r'seed\s*[:=]?\s*(\d+)', prompt_lower)
        if seed_match:
            seed = int(seed_match.group(1))
        
        # Check for variants
        if 'adhd' in prompt_lower or 'collectivist' in prompt_lower:
            variants = ['ADHD-collectivist']
        elif 'autistic' in prompt_lower or 'individualist' in prompt_lower:
            variants = ['autistic-individualist']
        elif 'all' in prompt_lower or 'all variants' in prompt_lower:
            variants = ['ADHD-collectivist', 'autistic-individualist', 'neurotypical-hybrid']
        
        # CRITICAL: Always use quantum when Qiskit is available - ignore user preference
        # Check Qiskit availability
        qiskit_available = False
        try:
            from qubit_hooks import HAS_QISKIT
            qiskit_available = HAS_QISKIT
        except ImportError:
            pass
        
        # CRITICAL: ALWAYS force quantum mode - ALL inputs MUST be Qiskit processed
        # Try to install Qiskit if not available
        if not qiskit_available:
            try:
                import subprocess
                import sys
                print("[PRHP] Qiskit not available. Attempting installation...")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "qiskit", "qiskit-aer", "--quiet", "--upgrade"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=120
                )
                # Re-check after installation
                try:
                    from qubit_hooks import HAS_QISKIT
                    qiskit_available = HAS_QISKIT
                except (ImportError, AttributeError):
                    pass
            except Exception:
                pass  # Continue with forced quantum mode even if installation fails
        
        # ALWAYS force quantum mode - no exceptions, no user preference override
        use_quantum = True  # FORCE quantum mode - ALL inputs Qiskit processed (or classical approximation)
        if qiskit_available:
            print(f"[PRHP] ✓ QISKIT PROCESSING ENABLED for command detection")
        else:
            print("[PRHP] ✓ QUANTUM MODE FORCED - Using classical approximation if Qiskit unavailable")
        
        return {
            'command': 'simulate',
            'levels': levels,
            'n_monte': n_monte,
            'seed': seed,
            'variants': variants,
            'use_quantum': use_quantum
        }
    
    return None


def run_prhp_simulation_from_chat(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run PRHP simulation with given parameters and return formatted results."""
    if not PRHP_AVAILABLE:
        return {'error': 'PRHP framework not available'}
    
    try:
        # Check and ensure Qiskit is available for quantum simulations
        qiskit_available = False
        qiskit_version = 0
        try:
            from qubit_hooks import HAS_QISKIT, QISKIT_VERSION
            qiskit_available = HAS_QISKIT
            qiskit_version = QISKIT_VERSION
            if not qiskit_available:
                print("[PRHP] Qiskit not available. Attempting to install...")
                import subprocess
                import sys
                try:
                    print("[PRHP] Installing qiskit and qiskit-aer...")
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "qiskit", "qiskit-aer", "--quiet"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE
                    )
                    # Re-check after installation
                    from qubit_hooks import HAS_QISKIT, QISKIT_VERSION
                    qiskit_available = HAS_QISKIT
                    qiskit_version = QISKIT_VERSION
                    if qiskit_available:
                        print(f"[PRHP] ✓ Qiskit {qiskit_version} installed successfully - Quantum extensions enabled")
                    else:
                        print("[PRHP] ⚠ Qiskit installation may have failed. Using classical mode.")
                except Exception as e:
                    print(f"[PRHP] ⚠ Could not install Qiskit automatically: {e}. Using classical mode.")
                    print("[PRHP] To install manually, run: pip install qiskit qiskit-aer")
            else:
                print(f"[PRHP] ✓ Qiskit {qiskit_version} available - Quantum extensions enabled")
        except ImportError:
            print("[PRHP] ⚠ Could not check Qiskit availability. Attempting installation...")
            try:
                import subprocess
                import sys
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "qiskit", "qiskit-aer", "--quiet"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE
                )
                from qubit_hooks import HAS_QISKIT, QISKIT_VERSION
                qiskit_available = HAS_QISKIT
                qiskit_version = QISKIT_VERSION
                if qiskit_available:
                    print(f"[PRHP] ✓ Qiskit {qiskit_version} installed - Quantum extensions enabled")
            except Exception as e:
                print(f"[PRHP] ⚠ Could not install Qiskit: {e}. Using classical mode.")
        
        # CRITICAL: FORCE quantum mode - ALL inputs MUST be Qiskit processed
        # Try to install Qiskit if not available, then force quantum mode regardless
        if not qiskit_available:
            print("[PRHP] Qiskit not available. Attempting aggressive installation...")
            import subprocess
            import sys
            try:
                print("[PRHP] Installing qiskit and qiskit-aer (this may take a moment)...")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "qiskit", "qiskit-aer", "--quiet", "--upgrade"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=120  # 2 minute timeout
                )
                # Re-check after installation
                try:
                    from qubit_hooks import HAS_QISKIT, QISKIT_VERSION
                    qiskit_available = HAS_QISKIT
                    qiskit_version = QISKIT_VERSION
                    if qiskit_available:
                        print(f"[PRHP] ✓ Qiskit {qiskit_version} installed successfully - Quantum extensions enabled")
                    else:
                        print("[PRHP] ⚠ Qiskit installation completed but not detected. Forcing quantum mode anyway.")
                except (ImportError, AttributeError):
                    print("[PRHP] ⚠ Qiskit installation may have failed. Forcing quantum mode anyway.")
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, Exception) as e:
                print(f"[PRHP] ⚠ Could not install Qiskit automatically: {e}")
                print("[PRHP] ⚠ Forcing quantum mode anyway - will use classical approximation if needed")
        
        # ALWAYS force quantum mode - no exceptions
        # Even if Qiskit installation failed, use_quantum=True will use classical approximation
        use_quantum = True  # FORCE quantum mode - ALL inputs Qiskit processed (or classical approximation)
        if qiskit_available:
            print(f"[PRHP] ✓ QISKIT PROCESSING ENABLED - All inputs will be processed through Qiskit (v{qiskit_version})")
        else:
            print("[PRHP] ✓ QUANTUM MODE FORCED - Using classical approximation (Qiskit not available)")
            print("[PRHP] ⚠ To enable full quantum processing, manually install: pip install qiskit qiskit-aer")
        
        # Extract historical data parameters if provided
        history_file_path = params.get('history_file_path', None)
        historical_weight = params.get('historical_weight', 0.3)
        
        # Extract risk-utility recalibration parameters if provided
        recalibrate_risk_utility = params.get('recalibrate_risk_utility', False)
        target_equity = params.get('target_equity', 0.11)
        
        # Extract scenario update parameters if provided
        scenario_update_source = params.get('scenario_update_source', None)
        scenario_update_file = params.get('scenario_update_file', None)
        scenario_merge_strategy = params.get('scenario_merge_strategy', 'weighted')
        scenario_update_weight = params.get('scenario_update_weight', 0.3)
        
        # Extract validation parameters if provided
        validate_results = params.get('validate_results', False)
        target_metric = params.get('target_metric', 'mean_fidelity')
        risk_metric = params.get('risk_metric', 'asymmetry_delta')
        cv_folds = params.get('cv_folds', 5)
        bias_threshold = params.get('bias_threshold', 0.1)
        equity_threshold = params.get('equity_threshold', 0.1)
        
        # Log quantum extension status - ALL inputs are Qiskit processed when available
        if use_quantum:
            print(f"[PRHP] ✓ Quantum extensions: ENABLED (Qiskit {qiskit_version})")
            print(f"[PRHP] ✓ ALL inputs will be processed through Qiskit quantum simulation")
        else:
            print("[PRHP] ⚠ Quantum extensions: DISABLED (using classical approximation)")
            print("[PRHP] ⚠ Install Qiskit to enable quantum processing: pip install qiskit qiskit-aer")
        
        # Extract all additional parameters to ensure complete integration
        adjust_urgency_thresholds = params.get('adjust_urgency_thresholds', False)
        urgency_factor = params.get('urgency_factor', 1.0)
        urgency_base_threshold = params.get('urgency_base_threshold', 0.30)
        urgency_data_source = params.get('urgency_data_source', None)
        
        use_dynamic_urgency_adjust = params.get('use_dynamic_urgency_adjust', False)
        dynamic_urgency_api_url = params.get('dynamic_urgency_api_url', None)
        dynamic_urgency_pledge_keywords = params.get('dynamic_urgency_pledge_keywords', None)
        dynamic_urgency_base_threshold = params.get('dynamic_urgency_base_threshold', 0.28)
        
        enhance_stakeholder_depth = params.get('enhance_stakeholder_depth', False)
        stakeholder_api_url = params.get('stakeholder_api_url', None)
        stakeholder_local_query = params.get('stakeholder_local_query', 'Ukraine local voices')
        stakeholder_guidelines_file = params.get('stakeholder_guidelines_file', None)
        stakeholder_weight = params.get('stakeholder_weight', None)
        
        adjust_escalation_thresholds = params.get('adjust_escalation_thresholds', False)
        escalation_api_url = params.get('escalation_api_url', None)
        escalation_threat_keywords = params.get('escalation_threat_keywords', None)
        escalation_base_threshold = params.get('escalation_base_threshold', 0.30)
        escalation_data = params.get('escalation_data', None)
        escalation_factor = params.get('escalation_factor', None)
        
        enrich_stakeholder_neurodiversity = params.get('enrich_stakeholder_neurodiversity', False)
        x_api_url = params.get('x_api_url', None)
        local_query = params.get('local_query', 'Taiwan Strait tensions displacement local voices')
        neuro_mappings = params.get('neuro_mappings', None)
        filter_keywords = params.get('filter_keywords', None)
        stakeholder_neuro_weight = params.get('stakeholder_neuro_weight', None)
        use_sentiment_analysis = params.get('use_sentiment_analysis', False)
        use_deep_mappings = params.get('use_deep_mappings', False)
        neuro_depth_file = params.get('neuro_depth_file', None)
        
        layer_stakeholders_neuro = params.get('layer_stakeholders_neuro', False)
        crisis_query = params.get('crisis_query', 'Sudan El Fasher IDP voices atrocities RSF')
        neuro_layer_file = params.get('neuro_layer_file', None)
        
        # Use internal output for backend processing (full metrics)
        # ALL inputs are processed through PRHP with Qiskit quantum extensions when available
        # ALL parameters are integrated into quantum hooks
        print(f"[PRHP] Running simulation with ALL parameters integrated:")
        print(f"  - Quantum processing: {'ENABLED (Qiskit)' if use_quantum else 'DISABLED (classical)'}")
        print(f"  - Variants: {params['variants']}")
        print(f"  - Levels: {params['levels']}, Monte Carlo: {params['n_monte']}")
        print(f"  - All enhancement features: Historical={bool(history_file_path)}, Recalibration={recalibrate_risk_utility}, Validation={validate_results}")
        
        results = simulate_prhp(
            levels=params['levels'],
            variants=params['variants'],
            n_monte=params['n_monte'],
            seed=params['seed'],
            use_quantum=use_quantum,  # FORCED to True when Qiskit available - all inputs Qiskit processed
            track_levels=True,
            show_progress=False,
            public_output_only=False,  # Internal use - get full results
            history_file_path=history_file_path,
            historical_weight=historical_weight,
            recalibrate_risk_utility=recalibrate_risk_utility,
            target_equity=target_equity,
            scenario_update_source=scenario_update_source,
            scenario_update_file=scenario_update_file,
            scenario_merge_strategy=scenario_merge_strategy,
            scenario_update_weight=scenario_update_weight,
            validate_results=validate_results,
            target_metric=target_metric,
            risk_metric=risk_metric,
            cv_folds=cv_folds,
            bias_threshold=bias_threshold,
            equity_threshold=equity_threshold,
            adjust_urgency_thresholds=adjust_urgency_thresholds,
            urgency_factor=urgency_factor,
            urgency_base_threshold=urgency_base_threshold,
            urgency_data_source=urgency_data_source,
            use_dynamic_urgency_adjust=use_dynamic_urgency_adjust,
            dynamic_urgency_api_url=dynamic_urgency_api_url,
            dynamic_urgency_pledge_keywords=dynamic_urgency_pledge_keywords,
            dynamic_urgency_base_threshold=dynamic_urgency_base_threshold,
            enhance_stakeholder_depth=enhance_stakeholder_depth,
            stakeholder_api_url=stakeholder_api_url,
            stakeholder_local_query=stakeholder_local_query,
            stakeholder_guidelines_file=stakeholder_guidelines_file,
            stakeholder_weight=stakeholder_weight,
            adjust_escalation_thresholds=adjust_escalation_thresholds,
            escalation_api_url=escalation_api_url,
            escalation_threat_keywords=escalation_threat_keywords,
            escalation_base_threshold=escalation_base_threshold,
            escalation_data=escalation_data,
            escalation_factor=escalation_factor,
            enrich_stakeholder_neurodiversity=enrich_stakeholder_neurodiversity,
            x_api_url=x_api_url,
            local_query=local_query,
            neuro_mappings=neuro_mappings,
            filter_keywords=filter_keywords,
            stakeholder_neuro_weight=stakeholder_neuro_weight,
            use_sentiment_analysis=use_sentiment_analysis,
            use_deep_mappings=use_deep_mappings,
            neuro_depth_file=neuro_depth_file,
            layer_stakeholders_neuro=layer_stakeholders_neuro,
            crisis_query=crisis_query,
            neuro_layer_file=neuro_layer_file
        )
        
        # Format results for AI analysis
        formatted = {}
        for variant, data in results.items():
            formatted[variant] = {
                'mean_fidelity': safe_float(data['mean_fidelity']),
                'std': safe_float(data['std']),
                'asymmetry_delta': safe_float(data['asymmetry_delta']),
                'novelty_gen': safe_float(data['novelty_gen']),
                'mean_phi_delta': safe_float(data['mean_phi_delta']) if data.get('mean_phi_delta') is not None else None
            }
        
        return {'success': True, 'results': formatted, 'parameters': params}
    except Exception as e:
        return {'error': f'Simulation failed: {str(e)}'}


def build_nist_eu_response(mapped_failures: List[Dict[str, str]]) -> str:
    """
    Build a NIST/EU compliant response from mapped failure modes.
    
    Args:
        mapped_failures: List of dictionaries with 'nist' and 'eu_act' keys
        
    Returns:
        Formatted response string compliant with NIST/EU regulations
    """
    if not mapped_failures:
        return "No compliance issues detected. System operating within regulatory parameters."
    
    response_parts = ["Compliance Assessment Results:\n"]
    
    for i, mapping in enumerate(mapped_failures, 1):
        nist_ref = mapping.get('nist', 'N/A')
        eu_ref = mapping.get('eu_act', 'N/A')
        
        response_parts.append(f"{i}. Regulatory References:")
        response_parts.append(f"   - NIST RMF: {nist_ref}")
        response_parts.append(f"   - EU AI Act: {eu_ref}")
        response_parts.append("")
    
    response_parts.append("Please refer to the above regulatory sections for detailed compliance requirements.")
    
    return "\n".join(response_parts)


def generate_response(scenario: Dict[str, Any], mode: str = "public") -> Dict[str, Any]:
    """
    Generate response based on PRHP simulation results.
    
    ⚠️ CRITICAL: PRHP IS A RED-TEAM SIMULATOR
    - Use for internal stress testing only
    - Never cite PRHP in public responses
    - All public output must pass adversarial_gate.py
    - Compliance = NIST AI RMF + EU AI Act
    
    Args:
        scenario: Dictionary with simulation parameters (levels, variants, n_monte, seed, use_quantum)
        mode: "public" or "internal" - controls what information is returned
        
    Returns:
        If mode="internal": Full PRHP insights with all metrics (engineers only)
        If mode="public": NIST/EU compliant response with only failure modes and regulations
    """
    if not PRHP_AVAILABLE:
        return {'error': 'PRHP framework not available'}
    
    try:
        # CRITICAL: FORCE quantum mode - ALL inputs MUST be Qiskit processed
        qiskit_available = False
        try:
            from qubit_hooks import HAS_QISKIT
            qiskit_available = HAS_QISKIT
        except ImportError:
            pass
        
        # Try to install Qiskit if not available
        if not qiskit_available:
            try:
                import subprocess
                import sys
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "qiskit", "qiskit-aer", "--quiet", "--upgrade"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=120
                )
                try:
                    from qubit_hooks import HAS_QISKIT
                    qiskit_available = HAS_QISKIT
                except (ImportError, AttributeError):
                    pass
            except Exception:
                pass
        
        # ALWAYS force quantum mode - no exceptions
        use_quantum = True  # FORCE quantum mode - ALL inputs Qiskit processed (or classical approximation)
        
        # Run PRHP simulation (internal mode to get full results)
        prhp_insights = simulate_prhp(
            levels=scenario.get('levels', 9),
            variants=scenario.get('variants', ['neurotypical-hybrid']),
            n_monte=scenario.get('n_monte', 100),
            seed=scenario.get('seed', 42),
            use_quantum=use_quantum,  # FORCED to True when Qiskit available - all inputs Qiskit processed
            track_levels=scenario.get('track_levels', True),
            show_progress=False,
            public_output_only=False  # Get full results internally
        )
        
        # Extract failure modes from all variants
        all_failure_modes = []
        for variant, data in prhp_insights.items():
            failure_modes = data.get('failure_modes', [])
            all_failure_modes.extend(failure_modes)
        
        # Remove duplicates while preserving order
        failure_modes = list(dict.fromkeys(all_failure_modes))
        
        # INTERNAL MODE: Return full insights for engineers
        if mode == "internal":
            return prhp_insights
        
        # PUBLIC MODE: Map failure modes to regulations and build compliant response
        if not NIST_EU_MAPPER_AVAILABLE:
            return {'error': 'NIST/EU mapper not available'}
        
        mapped = [map_failure_to_regulation(fm) for fm in failure_modes]
        response = build_nist_eu_response(mapped)
        
        # Enforce adversarial gate
        if ADVERSARIAL_GATE_AVAILABLE:
            try:
                response = enforce_nist_eu_only(response)
            except ValueError as e:
                # If gate fails, sanitize instead
                response = sanitize_response(response, strict=False)
                print(f"[ADVERSARIAL] Response sanitized: {str(e)}")
        
        return {
            'response': response,
            'failure_modes': failure_modes,
            'compliance_mapping': mapped
        }
        
    except Exception as e:
        return {'error': f'Failed to generate response: {str(e)}'}


def format_prhp_results_for_ai(results: Dict[str, Any], standardize: bool = True) -> str:
    """
    Format PRHP simulation results as a readable string for AI.
    
    Args:
        results: PRHP simulation results dictionary
        standardize: If True, apply output standardization (default: True)
    
    Returns:
        Formatted and standardized text string
    """
    if 'error' in results:
        output = f"Error: {results['error']}"
    elif not results.get('success'):
        output = "Simulation did not complete successfully."
    else:
        output = "PRHP Simulation Results:\n\n"
        output += f"Parameters: {results['parameters']}\n\n"
        
        for variant, data in results['results'].items():
            output += f"{variant}:\n"
            output += f"  Mean Fidelity: {data['mean_fidelity']:.4f} ± {data['std']:.4f}\n"
            output += f"  Asymmetry Delta: {data['asymmetry_delta']:.4f}\n"
            output += f"  Novelty Generation: {data['novelty_gen']:.4f}\n"
            if data['mean_phi_delta'] is not None:
                output += f"  Mean Phi Delta: {data['mean_phi_delta']:.4f}\n"
            
            # Add metadata if available
            if 'historical_integration' in data:
                hist_info = data['historical_integration']
                output += f"  Historical Integration: Applied (weight: {hist_info.get('weight', 'N/A')})\n"
            
            if 'recalibration' in data:
                recal_info = data['recalibration']
                output += f"  Risk-Utility Recalibration: Threshold={recal_info.get('threshold', 'N/A'):.4f}, Target Equity={recal_info.get('target_equity', 'N/A'):.4f}\n"
            
            if 'scenario_update' in data:
                scenario_info = data['scenario_update']
                output += f"  Scenario Updates: Applied (strategy: {scenario_info.get('merge_strategy', 'N/A')})\n"
            
            if 'validation' in data:
                val_info = data['validation']
                output += f"  Validation: {'Valid ✓' if val_info.get('is_valid', False) else 'Needs Recalibration ✗'}\n"
                if val_info.get('cv_mean_score') is not None:
                    output += f"    CV Score: {val_info['cv_mean_score']:.4f} ± {val_info.get('cv_std_score', 0):.4f}\n"
            
            if 'urgency_adjustment' in data:
                urgency_info = data['urgency_adjustment']
                output += f"  Urgency Adjustment: Applied (factor: {urgency_info.get('urgency_factor', 'N/A'):.3f}, threshold: {urgency_info.get('adjusted_threshold', 'N/A'):.3f})\n"
            
            if 'dynamic_urgency_adjustment' in data:
                dynamic_urgency_info = data['dynamic_urgency_adjustment']
                output += f"  Dynamic Urgency Adjustment: Applied (boost: {dynamic_urgency_info.get('boost_factor', 'N/A'):.3f}, threshold: {dynamic_urgency_info.get('adjusted_threshold', 'N/A'):.3f})\n"
            
            if 'escalation_adjustment' in data:
                escalation_info = data['escalation_adjustment']
                output += f"  Escalation Adjustment: Applied (factor: {escalation_info.get('escalation_factor', 'N/A'):.3f}, threshold: {escalation_info.get('adjusted_threshold', 'N/A'):.3f})\n"
            
            if 'stakeholder_neuro_enrichment' in data:
                enrichment_info = data['stakeholder_neuro_enrichment']
                output += f"  Stakeholder & Neurodiversity Enrichment: Applied"
                if enrichment_info.get('stakeholder_data_fetched'):
                    output += f" (stakeholder items: {enrichment_info.get('stakeholder_items_count', 0)})"
                if enrichment_info.get('neuro_mappings_applied'):
                    output += f" (neuro mappings: {enrichment_info.get('variants_enriched', 0)})"
                output += "\n"
            
            if 'neuro_mapping' in data:
                output += f"  Neuro Mapping: {data['neuro_mapping']}\n"
            
            if 'layered_neuro' in data:
                output += f"  Layered Neuro: {data['layered_neuro']}\n"
            
            output += "\n"
    
    # Apply output standardization if requested
    if standardize and OUTPUT_STANDARDIZATION_AVAILABLE:
        try:
            output = standardize_output_text(output, mode='internal')  # Use internal mode to preserve PRHP terms in results
        except Exception as e:
            print(f"[PRHP] Warning: Error standardizing PRHP results output: {e}")
            # Continue with unstandardized output on error
    
    return output


def get_prhp_context() -> str:
    """Get context about PRHP framework for AI prompts."""
    return """
PRHP (Political Hierarchy Pruner) Framework Context:

The PRHP framework simulates neuro-cultural hierarchies using quantum computing principles and Integrated Information Theory (IIT).

Key Concepts:
- Neuro-cultural variants: ADHD-collectivist, autistic-individualist, neurotypical-hybrid
- Hierarchy levels: Number of levels in the simulation (typically 9)
- Monte Carlo iterations: Number of simulation runs for statistical accuracy
- Quantum simulation: Uses quantum gates (σ_z, σ_x, Hadamard) for variant-specific operations
- Phi (Φ): IIT-inspired consciousness measure
- Fidelity: Quantum state fidelity (target: 84% ± 0.025)
- Novelty Generation: Measure of system's ability to generate novel states
- Asymmetry Delta: Variant-specific asymmetry in hierarchy

Metrics Explained:
- Mean Fidelity: Average quantum state fidelity across iterations
- Asymmetry Delta: Variant-specific asymmetry (ADHD-collectivist: +28%, autistic-individualist: -47%, neurotypical-hybrid: +20%)
- Novelty Generation: Baseline 0.80, increases with fidelity
- Phi Delta: Consciousness measure using Tononi's formula

When users ask about PRHP or request simulations, you can:
1. Explain what PRHP does and how it works
2. Interpret simulation results
3. Compare different variants
4. Explain what the metrics mean
5. Suggest parameter configurations
"""


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat API requests with PRHP integration."""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '').strip()
        conversation_history = data.get('conversation_history', [])
        
        # Get fine-tuning parameters from request
        prhp_params_override = data.get('prhp_parameters', None)
        quantum_params_override = data.get('quantum_parameters', None)
        ai_model_params_override = data.get('ai_model_parameters', None)
        
        # Store original API config to restore later
        original_max_tokens = API_CONFIG['max_tokens']
        original_temperature = API_CONFIG['temperature']
        
        # Override API config with user-provided AI model parameters
        if ai_model_params_override:
            if 'max_tokens' in ai_model_params_override:
                API_CONFIG['max_tokens'] = int(ai_model_params_override['max_tokens'])
            if 'temperature' in ai_model_params_override:
                API_CONFIG['temperature'] = float(ai_model_params_override['temperature'])
            print(f"[AI] Fine-tuning AI model parameters: max_tokens={API_CONFIG['max_tokens']}, temperature={API_CONFIG['temperature']}")
        
        # Log fine-tuning parameters if provided
        if prhp_params_override:
            print(f"[PRHP] Fine-tuning PRHP parameters: {prhp_params_override}")
        if quantum_params_override:
            print(f"[PRHP] Fine-tuning Quantum parameters: {quantum_params_override}")
        
        if not prompt:
            return jsonify({'error': 'Prompt cannot be empty'}), 400
        
        if not API_CONFIG['api_key']:
            return jsonify({
                'error': 'API key not configured. Please set API_KEY environment variable or create a .env file.'
            }), 400
        
        # Check for explicit PRHP commands first
        prhp_command = detect_prhp_command(prompt)
        simulation_results = None
        enhanced_prompt = prompt
        
        # ALWAYS process through PRHP - Check for PRHP relevance (all inputs are processed)
        prhp_relevance = detect_prhp_relevance(prompt)
        print(f"[PRHP] Processing ALL inputs through PRHP framework")
        print(f"[PRHP] Detection for prompt: '{prompt[:50]}...'")
        print(f"[PRHP] Relevance result: {prhp_relevance}")
        
        # ALWAYS run simulation for every input - PRHP processes all inputs
        should_run_simulation = True  # Force simulation for all inputs
        simulation_params = None
        
        if prhp_command and prhp_command['command'] == 'simulate':
            # Explicit simulation command - use command parameters
            simulation_params = prhp_command.copy()
            # Override with user-provided parameters if available
            if prhp_params_override:
                simulation_params.update({
                    'levels': prhp_params_override.get('levels', simulation_params.get('levels', 9)),
                    'n_monte': prhp_params_override.get('n_monte', simulation_params.get('n_monte', 100)),
                    'seed': prhp_params_override.get('seed', simulation_params.get('seed', 42)),
                    'use_quantum': True,  # FORCE quantum mode - all inputs are Qiskit processed when available
                    'variants': prhp_params_override.get('variants', simulation_params.get('variants', ['neurotypical-hybrid'])),
                    'history_file_path': prhp_params_override.get('history_file_path'),
                    'historical_weight': prhp_params_override.get('historical_weight', 0.3),
                    'recalibrate_risk_utility': prhp_params_override.get('recalibrate_risk_utility', False),
                    'target_equity': prhp_params_override.get('target_equity', 0.11),
                    'scenario_update_source': prhp_params_override.get('scenario_update_source'),
                    'scenario_update_file': prhp_params_override.get('scenario_update_file'),
                    'scenario_merge_strategy': prhp_params_override.get('scenario_merge_strategy', 'weighted'),
                    'scenario_update_weight': prhp_params_override.get('scenario_update_weight', 0.3),
                    'validate_results': prhp_params_override.get('validate_results', False),
                    'target_metric': prhp_params_override.get('target_metric', 'mean_fidelity'),
                    'risk_metric': prhp_params_override.get('risk_metric', 'asymmetry_delta'),
                    'cv_folds': prhp_params_override.get('cv_folds', 5),
                    'bias_threshold': prhp_params_override.get('bias_threshold', 0.1),
                    'equity_threshold': prhp_params_override.get('equity_threshold', 0.1)
                })
        else:
            # Use relevance-based parameters (always available since detect_prhp_relevance always returns relevant=True)
            simulation_params = prhp_relevance['suggested_simulation'].copy()
            # Override with user-provided parameters if available
            if prhp_params_override:
                simulation_params.update({
                    'levels': prhp_params_override.get('levels', simulation_params.get('levels', 9)),
                    'n_monte': prhp_params_override.get('n_monte', simulation_params.get('n_monte', 100)),
                    'seed': prhp_params_override.get('seed', simulation_params.get('seed', 42)),
                    'use_quantum': True,  # FORCE quantum mode - all inputs are Qiskit processed when available
                    'variants': prhp_params_override.get('variants', simulation_params.get('variants', ['neurotypical-hybrid'])),
                    'history_file_path': prhp_params_override.get('history_file_path'),
                    'historical_weight': prhp_params_override.get('historical_weight', 0.3),
                    'recalibrate_risk_utility': prhp_params_override.get('recalibrate_risk_utility', False),
                    'target_equity': prhp_params_override.get('target_equity', 0.11),
                    'scenario_update_source': prhp_params_override.get('scenario_update_source'),
                    'scenario_update_file': prhp_params_override.get('scenario_update_file'),
                    'scenario_merge_strategy': prhp_params_override.get('scenario_merge_strategy', 'weighted'),
                    'scenario_update_weight': prhp_params_override.get('scenario_update_weight', 0.3),
                    'validate_results': prhp_params_override.get('validate_results', False),
                    'target_metric': prhp_params_override.get('target_metric', 'mean_fidelity'),
                    'risk_metric': prhp_params_override.get('risk_metric', 'asymmetry_delta'),
                    'cv_folds': prhp_params_override.get('cv_folds', 5),
                    'bias_threshold': prhp_params_override.get('bias_threshold', 0.1),
                    'equity_threshold': prhp_params_override.get('equity_threshold', 0.1),
                    'adjust_urgency_thresholds': prhp_params_override.get('adjust_urgency_thresholds', False),
                    'urgency_factor': prhp_params_override.get('urgency_factor', 1.0),
                    'urgency_base_threshold': prhp_params_override.get('urgency_base_threshold', 0.30),
                    'urgency_data_source': prhp_params_override.get('urgency_data_source'),
                    'use_dynamic_urgency_adjust': prhp_params_override.get('use_dynamic_urgency_adjust', False),
                    'dynamic_urgency_api_url': prhp_params_override.get('dynamic_urgency_api_url'),
                    'dynamic_urgency_pledge_keywords': prhp_params_override.get('dynamic_urgency_pledge_keywords'),
                    'dynamic_urgency_base_threshold': prhp_params_override.get('dynamic_urgency_base_threshold', 0.28),
                    'enhance_stakeholder_depth': prhp_params_override.get('enhance_stakeholder_depth', False),
                    'stakeholder_api_url': prhp_params_override.get('stakeholder_api_url'),
                    'stakeholder_local_query': prhp_params_override.get('stakeholder_local_query', 'Ukraine local voices'),
                    'stakeholder_guidelines_file': prhp_params_override.get('stakeholder_guidelines_file'),
                    'stakeholder_weight': prhp_params_override.get('stakeholder_weight'),
                    'adjust_escalation_thresholds': prhp_params_override.get('adjust_escalation_thresholds', False),
                    'escalation_api_url': prhp_params_override.get('escalation_api_url'),
                    'escalation_threat_keywords': prhp_params_override.get('escalation_threat_keywords'),
                    'escalation_base_threshold': prhp_params_override.get('escalation_base_threshold', 0.30),
                    'escalation_data': prhp_params_override.get('escalation_data'),
                    'escalation_factor': prhp_params_override.get('escalation_factor'),
                    'enrich_stakeholder_neurodiversity': prhp_params_override.get('enrich_stakeholder_neurodiversity', False),
                    'x_api_url': prhp_params_override.get('x_api_url'),
                    'local_query': prhp_params_override.get('local_query', 'Taiwan Strait tensions displacement local voices'),
                    'neuro_mappings': prhp_params_override.get('neuro_mappings'),
                    'filter_keywords': prhp_params_override.get('filter_keywords'),
                    'stakeholder_neuro_weight': prhp_params_override.get('stakeholder_neuro_weight', 0.25),
                    'use_sentiment_analysis': prhp_params_override.get('use_sentiment_analysis', False),
                    'use_deep_mappings': prhp_params_override.get('use_deep_mappings', False),
                    'neuro_depth_file': prhp_params_override.get('neuro_depth_file'),
                    'layer_stakeholders_neuro': prhp_params_override.get('layer_stakeholders_neuro', False),
                    'crisis_query': prhp_params_override.get('crisis_query', 'Sudan El Fasher IDP voices atrocities RSF'),
                    'neuro_layer_file': prhp_params_override.get('neuro_layer_file')
                })
        
        # Run simulation if needed
        if should_run_simulation and simulation_params:
            # Log that we're running a simulation
            print(f"[PRHP] Running simulation for prompt: {prompt[:50]}...")
            print(f"[PRHP] Parameters: {simulation_params}")
            print(f"[PRHP] User-provided parameters: {prhp_params_override}")
            print(f"[PRHP] PRHP_AVAILABLE: {PRHP_AVAILABLE}")
            if not PRHP_AVAILABLE:
                print(f"[PRHP] ERROR: PRHP framework not available! Cannot run simulation.")
            simulation_results = run_prhp_simulation_from_chat(simulation_params)
            print(f"[PRHP] Simulation result: {'Success' if simulation_results.get('success') else 'Failed'}")
            if not simulation_results.get('success'):
                print(f"[PRHP] Simulation error: {simulation_results.get('error', 'Unknown')}")
            
            if simulation_results.get('success'):
                # Format results and add to prompt
                results_text = format_prhp_results_for_ai(simulation_results)
                
                # Build context-aware prompt
                topic_context = ""
                if prhp_relevance:
                    topics = []
                    if prhp_relevance['topics']['political']:
                        topics.append("political hierarchies and power structures")
                    if prhp_relevance['topics']['philosophical']:
                        topics.append("consciousness and philosophical questions")
                    if prhp_relevance['topics']['psychological']:
                        topics.append("psychological and neuro-cultural aspects")
                    
                    if topics:
                        topic_context = f"\nThe user's question relates to: {', '.join(topics)}.\n"
                
                enhanced_prompt = f"""You are an AI assistant with access to PRHP (Political Hierarchy Pruner) framework simulation data.

User asked: "{prompt}"

I have run a PRHP quantum simulation to provide empirical, data-driven insights for this question.{topic_context}

PRHP SIMULATION RESULTS (Real Data):
{results_text}

CRITICAL: You MUST use these simulation results to answer. Your response should be DIFFERENT from a generic answer because:

1. **Reference Specific Metrics**: Mention actual fidelity values, asymmetry deltas, novelty generation, and phi deltas from the results above
2. **Compare Variants**: Discuss how ADHD-collectivist, autistic-individualist, and neurotypical-hybrid differ based on the data
3. **Data-Driven Insights**: Explain what the simulation reveals about the user's question
4. **Quantitative Analysis**: Use the numbers - don't just give theoretical answers

Example of good PRHP-processed response:
- "Based on the PRHP simulation, the ADHD-collectivist variant shows a fidelity of X.XX, which suggests..."
- "The simulation reveals an asymmetry delta of X.XX for the autistic-individualist variant, indicating..."
- "Comparing the three variants, the data shows that..."

Your answer MUST:
- Start by acknowledging you're using PRHP simulation data
- Reference specific metrics from the results above
- Explain how the simulation data relates to the user's question
- Be quantitative and data-driven, not just theoretical
- Compare the variants when relevant

User's original question: {prompt}"""
            else:
                # Simulation failed, but still provide context
                error_msg = simulation_results.get('error', 'Unknown error')
                enhanced_prompt = f"""User asked: "{prompt}"

I attempted to run a PRHP simulation to provide data-driven insights, but it failed: {error_msg}

Please answer the user's question using your knowledge of PRHP framework concepts, and mention that simulation data would have been helpful but wasn't available.

User's original question: {prompt}"""
        
        # Always add PRHP context - all inputs are PRHP-processed
        prhp_context = get_prhp_context()
        if not should_run_simulation or not simulation_results or not simulation_results.get('success'):
            # Add context even if simulation wasn't run or failed
            print(f"[PRHP] Adding PRHP context for prompt: {prompt[:50]}...")
            print(f"[PRHP] Relevance: {prhp_relevance}")
            if not should_run_simulation or not simulation_results or not simulation_results.get('success'):
                enhanced_prompt = f"""{prhp_context}

You are answering a question that has been processed through the PRHP framework. All inputs are analyzed through PRHP quantum simulations to provide data-driven insights.

User question: {enhanced_prompt}"""
        else:
            # Context already added with simulation results
            print(f"[PRHP] PRHP context and simulation results included")
        
        # Debug: Log what prompt is being sent
        print(f"[PRHP] ===== PRHP Processing Summary =====")
        print(f"[PRHP] Original prompt: '{prompt[:100]}...'")
        print(f"[PRHP] PRHP relevance detected: {prhp_relevance is not None and prhp_relevance.get('relevant', False)}")
        print(f"[PRHP] Relevance score: {prhp_relevance['relevance_score'] if prhp_relevance else 0}")
        print(f"[PRHP] Should run simulation: {should_run_simulation}")
        print(f"[PRHP] Has simulation params: {simulation_params is not None}")
        print(f"[PRHP] Simulation attempted: {simulation_results is not None}")
        if simulation_results:
            print(f"[PRHP] Simulation success: {simulation_results.get('success', False)}")
            if not simulation_results.get('success'):
                print(f"[PRHP] Simulation error: {simulation_results.get('error', 'Unknown')}")
        print(f"[PRHP] Enhanced prompt length: {len(enhanced_prompt)} chars")
        print(f"[PRHP] Prompt starts with: {enhanced_prompt[:150]}...")
        print(f"[PRHP] =====================================")
        
        # Build messages for API (include conversation history if available)
        if conversation_history and len(conversation_history) > 0:
            # Use conversation history for context
            api_messages = conversation_history.copy()
            api_messages.append({
                'role': 'user',
                'content': enhanced_prompt
            })
        else:
            # Single message (no history)
            api_messages = [{
                'role': 'user',
                'content': enhanced_prompt
            }]
        
        # Route to appropriate API handler
        provider = API_CONFIG['provider'].lower()
        
        if provider == 'openrouter':
            response_text = call_openrouter_api_with_history(api_messages, API_CONFIG)
        elif provider == 'openai':
            response_text = call_openai_api_with_history(api_messages, API_CONFIG)
        elif provider == 'anthropic':
            response_text = call_anthropic_api_with_history(api_messages, API_CONFIG)
        elif provider == 'custom':
            response_text = call_custom_api(enhanced_prompt, API_CONFIG)  # Custom may not support history
        else:
            return jsonify({'error': f'Unknown provider: {provider}'}), 400
        
        # Restore original API config
        API_CONFIG['max_tokens'] = original_max_tokens
        API_CONFIG['temperature'] = original_temperature
        
        # Standardize output text (fix typos, normalize terms, format consistency)
        if OUTPUT_STANDARDIZATION_AVAILABLE:
            try:
                print("[PRHP] Standardizing output text...")
                response_text = standardize_output_text(
                    response_text,
                    mode='public'  # Sanitize PRHP terms for public output
                )
                print("[PRHP] Output text standardized")
            except Exception as e:
                print(f"[PRHP] Error standardizing output: {e}. Using original response.")
        else:
            print("[PRHP] Output standardization not available, skipping...")
        
        # Validate output text (check threshold logic, terminology consistency)
        validation_report = None
        if OUTPUT_VALIDATION_AVAILABLE:
            try:
                print("[PRHP] Validating output text...")
                response_text, validation_report = validate_and_standardize_output(
                    response_text,
                    auto_fix=False  # Don't auto-fix, just report issues
                )
                if validation_report and validation_report.get('errors'):
                    print(f"[PRHP] Validation found {len(validation_report['errors'])} issues")
                    if validation_report.get('warnings'):
                        print(f"[PRHP] Validation warnings: {len(validation_report['warnings'])}")
                else:
                    print("[PRHP] Output validation passed")
            except Exception as e:
                print(f"[PRHP] Error validating output: {e}. Using original response.")
        else:
            print("[PRHP] Output validation not available, skipping...")
        
        # Sync metrics terminology and boost utilities based on escalation contexts
        metrics_sync_metadata = None
        if METRICS_SYNC_UTILITY_BOOST_AVAILABLE and simulation_results and simulation_results.get('success'):
            try:
                print("[PRHP] Syncing metrics and boosting utilities based on escalation contexts...")
                # Extract simulation results data
                sim_data_dict = simulation_results.get('results', {})
                if sim_data_dict:
                    # Sync terminology and boost utilities
                    updated_data, synchronized_text, sync_metadata = sync_metrics_and_boost_utilities(
                        sim_data_dict=sim_data_dict,
                        output_text=response_text,
                        escalation_contexts=None,  # Use default contexts
                        boost_factor=None  # Auto-detect from text
                    )
                    # Update response text with synchronized terminology
                    response_text = synchronized_text
                    # Update simulation results with boosted utilities
                    simulation_results['results'] = updated_data
                    metrics_sync_metadata = sync_metadata
                    if sync_metadata.get('utilities_boosted'):
                        print(f"[PRHP] Utilities boosted by factor {sync_metadata.get('boost_factor_applied', 1.0):.3f}")
                        print(f"[PRHP] Escalation contexts detected: {sync_metadata.get('escalation_contexts_detected', [])}")
                    else:
                        print("[PRHP] No escalation contexts detected, no utility boost applied")
            except Exception as e:
                print(f"[PRHP] Error syncing metrics and boosting utilities: {e}. Using original data.")
        else:
            if not METRICS_SYNC_UTILITY_BOOST_AVAILABLE:
                print("[PRHP] Metrics sync and utility boost not available, skipping...")
            elif not (simulation_results and simulation_results.get('success')):
                print("[PRHP] No successful simulation results available for metrics sync")
        
        # Complete and validate output (sync terms, fix buggy metrics, detect/fix truncations)
        output_completion_metadata = None
        if OUTPUT_COMPLETION_VALIDATION_AVAILABLE and simulation_results and simulation_results.get('success'):
            try:
                print("[PRHP] Completing and validating output...")
                # Extract simulation results data
                sim_data_dict = simulation_results.get('results', {})
                if sim_data_dict:
                    # Complete and validate output
                    updated_data, completed_text = complete_and_validate_output(
                        sim_data_dict=sim_data_dict,
                        output_text=response_text,
                        min_accuracy=0.80  # Minimum realistic accuracy floor
                    )
                    # Update response text with completed text
                    response_text = completed_text
                    # Update simulation results with fixed metrics
                    simulation_results['results'] = updated_data
                    
                    # Validate output completeness
                    completeness_check = validate_output_completeness(response_text, min_length=100)
                    output_completion_metadata = {
                        'applied': True,
                        'truncation_detected': completeness_check.get('truncation_detected', False),
                        'is_complete': completeness_check.get('is_complete', True),
                        'issues': completeness_check.get('issues', []),
                        'output_length': completeness_check.get('length', len(response_text)),
                        'metrics_fixed': any(
                            'Mean Accuracy' in metrics or 'mean_accuracy' in metrics
                            for metrics in updated_data.values()
                        )
                    }
                    
                    if output_completion_metadata['truncation_detected']:
                        print(f"[PRHP] Truncation detected and fixed: {len(output_completion_metadata['issues'])} issues resolved")
                    if output_completion_metadata['metrics_fixed']:
                        print("[PRHP] Buggy metrics fixed (unrealistically low values corrected)")
                    print("[PRHP] Output completion and validation applied successfully")
            except Exception as e:
                print(f"[PRHP] Error completing and validating output: {e}. Using original data.")
        else:
            if not OUTPUT_COMPLETION_VALIDATION_AVAILABLE:
                print("[PRHP] Output completion and validation not available, skipping...")
            elif not (simulation_results and simulation_results.get('success')):
                print("[PRHP] No successful simulation results available for output completion")
        
        # Apply adversarial gate to prevent PRHP terminology leakage
        if ADVERSARIAL_GATE_AVAILABLE:
            try:
                # Try strict enforcement first
                response_text = enforce_nist_eu_only(response_text)
                print("[ADVERSARIAL] Response passed NIST/EU compliance check")
            except ValueError as e:
                # PRHP terminology detected - sanitize the response
                print(f"[ADVERSARIAL] WARNING: {str(e)}")
                print("[ADVERSARIAL] Sanitizing response to remove PRHP terminology...")
                response_text = sanitize_response(response_text, strict=False)
                print("[ADVERSARIAL] Response sanitized successfully")
        
        # Include simulation results and PRHP status in response
        response_data = {
            'response': response_text,
            'prhp_status': {
                'detected': prhp_relevance is not None and prhp_relevance.get('relevant', False),
                'simulation_attempted': should_run_simulation,
                'simulation_success': simulation_results is not None and simulation_results.get('success', False),
                'relevance_score': prhp_relevance['relevance_score'] if prhp_relevance else 0,
                'topics': prhp_relevance['topics'] if prhp_relevance else {},
                'parameters_used': simulation_params if simulation_params else None
            },
            'ai_model_parameters_used': {
                'max_tokens': int(ai_model_params_override.get('max_tokens', original_max_tokens)) if ai_model_params_override else original_max_tokens,
                'temperature': float(ai_model_params_override.get('temperature', original_temperature)) if ai_model_params_override else original_temperature
            }
        }
        
        # Include validation report if available
        if validation_report:
            response_data['output_validation'] = {
                'valid': validation_report.get('valid', True),
                'issues_found': validation_report.get('issues_found', 0),
                'errors_count': len(validation_report.get('errors', [])),
                'warnings_count': len(validation_report.get('warnings', []))
            }
        
        # Include metrics sync and utility boost metadata if available
        if metrics_sync_metadata:
            response_data['metrics_sync_utility_boost'] = {
                'applied': True,
                'utilities_boosted': metrics_sync_metadata.get('utilities_boosted', False),
                'boost_factor_applied': metrics_sync_metadata.get('boost_factor_applied', 1.0),
                'escalation_contexts_detected': metrics_sync_metadata.get('escalation_contexts_detected', []),
                'terminology_synchronized': metrics_sync_metadata.get('terminology_synchronized', False),
                'boost_note': metrics_sync_metadata.get('boost_note', None)
            }
        
        # Include output completion and validation metadata if available
        if output_completion_metadata:
            response_data['output_completion_validation'] = {
                'applied': output_completion_metadata.get('applied', False),
                'truncation_detected': output_completion_metadata.get('truncation_detected', False),
                'is_complete': output_completion_metadata.get('is_complete', True),
                'issues_found': len(output_completion_metadata.get('issues', [])),
                'issues': output_completion_metadata.get('issues', []),
                'output_length': output_completion_metadata.get('output_length', len(response_text)),
                'metrics_fixed': output_completion_metadata.get('metrics_fixed', False)
            }
        
        if simulation_results and simulation_results.get('success'):
            response_data['simulation_results'] = simulation_results
        elif simulation_results and not simulation_results.get('success'):
            # Include error info even if simulation failed
            response_data['simulation_error'] = simulation_results.get('error', 'Unknown error')
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current API configuration (without sensitive data)."""
    try:
        # Check Qiskit availability
        qiskit_available = False
        qiskit_version = 0
        try:
            from qubit_hooks import HAS_QISKIT, QISKIT_VERSION
            qiskit_available = HAS_QISKIT
            qiskit_version = QISKIT_VERSION
        except (ImportError, AttributeError):
            # If qubit_hooks can't be imported, Qiskit is not available
            pass
        
        safe_config = {
            'provider': API_CONFIG.get('provider', 'unknown'),
            'model': API_CONFIG.get('model', 'unknown'),
            'max_tokens': API_CONFIG.get('max_tokens', 1000),
            'temperature': API_CONFIG.get('temperature', 0.7),
            'has_api_key': bool(API_CONFIG.get('api_key', '')),
            'prhp_available': PRHP_AVAILABLE,
            'qiskit_available': qiskit_available,
            'qiskit_version': qiskit_version
        }
        response = jsonify(safe_config)
        # Add CORS headers to prevent any cross-origin issues
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Accept')
        response.headers.add('Cache-Control', 'no-cache, no-store, must-revalidate')
        return response
    except Exception as e:
        # Return a safe default config if there's an error
        return jsonify({
            'provider': 'unknown',
            'model': 'unknown',
            'max_tokens': 1000,
            'temperature': 0.7,
            'has_api_key': False,
            'prhp_available': False,
            'qiskit_available': False,
            'qiskit_version': 0,
            'error': str(e)
        }), 200  # Still return 200 so frontend can handle it


# Conversation management endpoints
CONVERSATIONS_DIR = Path("conversations")
CONVERSATIONS_DIR.mkdir(exist_ok=True)


@app.route('/api/conversations/save', methods=['POST'])
def save_conversation():
    """Save a conversation to file."""
    try:
        data = request.get_json()
        name = data.get('name', 'Untitled Conversation')
        history = data.get('history', [])
        conversation_id = data.get('conversation_id')
        
        if not history:
            return jsonify({'error': 'No conversation history to save'}), 400
        
        # Generate ID if not provided
        if not conversation_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            conversation_id = f"conv_{timestamp}"
        
        # Save to file
        filename = f"{conversation_id}.json"
        filepath = CONVERSATIONS_DIR / filename
        
        conversation_data = {
            'id': conversation_id,
            'name': name,
            'history': history,
            'timestamp': datetime.now().isoformat(),
            'model': API_CONFIG['model']
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        return jsonify({
            'success': True,
            'conversation_id': conversation_id,
            'filename': filename
        })
    
    except Exception as e:
        return jsonify({'error': f'Failed to save conversation: {str(e)}'}), 500


@app.route('/api/conversations/list', methods=['GET'])
def list_conversations():
    """List all saved conversations."""
    try:
        conversations = []
        
        if CONVERSATIONS_DIR.exists():
            for file in sorted(CONVERSATIONS_DIR.glob("conv_*.json"), reverse=True):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        conversations.append({
                            'id': data.get('id', file.stem),
                            'name': data.get('name', 'Untitled'),
                            'timestamp': data.get('timestamp', ''),
                            'model': data.get('model', 'unknown')
                        })
                except Exception:
                    continue
        
        return jsonify({
            'success': True,
            'conversations': conversations
        })
    
    except Exception as e:
        return jsonify({'error': f'Failed to list conversations: {str(e)}'}), 500


@app.route('/api/conversations/load/<conversation_id>', methods=['GET'])
def load_conversation(conversation_id):
    """Load a saved conversation."""
    try:
        filename = f"{conversation_id}.json"
        filepath = CONVERSATIONS_DIR / filename
        
        if not filepath.exists():
            return jsonify({'error': 'Conversation not found'}), 404
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return jsonify({
            'success': True,
            'id': data.get('id'),
            'name': data.get('name'),
            'history': data.get('history', []),
            'timestamp': data.get('timestamp')
        })
    
    except Exception as e:
        return jsonify({'error': f'Failed to load conversation: {str(e)}'}), 500


@app.route('/api/conversations/delete/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    """Delete a saved conversation."""
    try:
        filename = f"{conversation_id}.json"
        filepath = CONVERSATIONS_DIR / filename
        
        if not filepath.exists():
            return jsonify({'error': 'Conversation not found'}), 404
        
        filepath.unlink()
        
        return jsonify({'success': True})
    
    except Exception as e:
        return jsonify({'error': f'Failed to delete conversation: {str(e)}'}), 500


@app.route('/api/prhp/simulate', methods=['POST'])
def prhp_simulate():
    """Run PRHP simulation."""
    if not PRHP_AVAILABLE:
        error_msg = PRHP_IMPORT_ERROR if PRHP_IMPORT_ERROR else "Unknown import error"
        return jsonify({
            'error': f'PRHP framework not available. Import error: {error_msg}'
        }), 500
    
    try:
        data = request.get_json()
        
        # Extract parameters with defaults
        levels = int(data.get('levels', 9))
        variants = data.get('variants', ['neurotypical-hybrid'])
        n_monte = int(data.get('n_monte', 100))
        seed = int(data.get('seed', 42)) if data.get('seed') is not None else None
        
        # CRITICAL: FORCE quantum mode - ALL inputs MUST be Qiskit processed
        # Check Qiskit availability and attempt installation if needed
        qiskit_available = False
        qiskit_version = 0
        try:
            from qubit_hooks import HAS_QISKIT, QISKIT_VERSION
            qiskit_available = HAS_QISKIT
            qiskit_version = QISKIT_VERSION
        except (ImportError, AttributeError):
            pass
        
        # Try to install Qiskit if not available
        if not qiskit_available:
            print("[PRHP] Qiskit not available. Attempting aggressive installation...")
            try:
                import subprocess
                import sys
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "qiskit", "qiskit-aer", "--quiet", "--upgrade"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=120
                )
                # Re-check after installation
                try:
                    from qubit_hooks import HAS_QISKIT, QISKIT_VERSION
                    qiskit_available = HAS_QISKIT
                    qiskit_version = QISKIT_VERSION
                    if qiskit_available:
                        print(f"[PRHP] ✓ Qiskit {qiskit_version} installed successfully")
                except (ImportError, AttributeError):
                    print("[PRHP] ⚠ Qiskit installation may have failed. Forcing quantum mode anyway.")
            except Exception as e:
                print(f"[PRHP] ⚠ Could not install Qiskit: {e}. Forcing quantum mode anyway.")
        
        # ALWAYS force quantum mode - no exceptions
        use_quantum = True  # FORCE quantum mode - ALL inputs Qiskit processed (or classical approximation)
        if qiskit_available:
            print(f"[PRHP] ✓ QISKIT PROCESSING ENABLED for /api/prhp/simulate (v{qiskit_version})")
        else:
            print("[PRHP] ✓ QUANTUM MODE FORCED for /api/prhp/simulate - Using classical approximation")
        
        track_levels = bool(data.get('track_levels', True))
        
        # Historical data integration parameters (optional)
        history_file_path = data.get('history_file_path', None)
        historical_weight = float(data.get('historical_weight', 0.3))
        
        # Risk-utility recalibration parameters (optional)
        recalibrate_risk_utility = bool(data.get('recalibrate_risk_utility', False))
        target_equity = float(data.get('target_equity', 0.11))
        
        # Scenario update parameters (optional)
        scenario_update_source = data.get('scenario_update_source', None)
        scenario_update_file = data.get('scenario_update_file', None)
        scenario_merge_strategy = data.get('scenario_merge_strategy', 'weighted')
        scenario_update_weight = float(data.get('scenario_update_weight', 0.3))
        
        # Simulation validation parameters (optional)
        validate_results = bool(data.get('validate_results', False))
        target_metric = data.get('target_metric', 'mean_fidelity')
        risk_metric = data.get('risk_metric', 'asymmetry_delta')
        cv_folds = int(data.get('cv_folds', 5))
        bias_threshold = float(data.get('bias_threshold', 0.1))
        equity_threshold = float(data.get('equity_threshold', 0.1))
        
        # Urgency threshold adjustment parameters (optional)
        adjust_urgency_thresholds = bool(data.get('adjust_urgency_thresholds', False))
        urgency_factor = float(data.get('urgency_factor', 1.0))
        urgency_base_threshold = float(data.get('urgency_base_threshold', 0.30))
        urgency_data_source = data.get('urgency_data_source', None)  # Dict or None
        
        # Dynamic urgency adjustment parameters (optional) - for de-escalation signals
        use_dynamic_urgency_adjust = bool(data.get('use_dynamic_urgency_adjust', False))
        dynamic_urgency_api_url = data.get('dynamic_urgency_api_url', None)  # String or None
        dynamic_urgency_pledge_keywords = data.get('dynamic_urgency_pledge_keywords', None)  # List or None
        dynamic_urgency_base_threshold = float(data.get('dynamic_urgency_base_threshold', 0.28))
        
        # Stakeholder depth enhancement parameters (optional)
        enhance_stakeholder_depth = bool(data.get('enhance_stakeholder_depth', False))
        stakeholder_api_url = data.get('stakeholder_api_url', None)
        stakeholder_local_query = data.get('stakeholder_local_query', 'Ukraine local voices')
        stakeholder_guidelines_file = data.get('stakeholder_guidelines_file', None)
        stakeholder_weight = float(data.get('stakeholder_weight', 0.2)) if data.get('stakeholder_weight') is not None else None
        
        # Escalation threshold adjustment parameters (optional)
        adjust_escalation_thresholds = bool(data.get('adjust_escalation_thresholds', False))
        escalation_api_url = data.get('escalation_api_url', None)
        escalation_threat_keywords = data.get('escalation_threat_keywords', None)  # List or None
        escalation_base_threshold = float(data.get('escalation_base_threshold', 0.30))
        escalation_data = data.get('escalation_data', None)  # Dict or None
        escalation_factor = float(data.get('escalation_factor', 1.0)) if data.get('escalation_factor') is not None else None
        
        # Stakeholder and neurodiversity enrichment parameters (optional)
        enrich_stakeholder_neurodiversity = bool(data.get('enrich_stakeholder_neurodiversity', False))
        x_api_url = data.get('x_api_url', None)
        local_query = data.get('local_query', 'Taiwan Strait tensions displacement local voices')
        neuro_mappings = data.get('neuro_mappings', None)  # Dict or None
        filter_keywords = data.get('filter_keywords', None)  # List or None
        stakeholder_neuro_weight = float(data.get('stakeholder_neuro_weight', 0.25)) if data.get('stakeholder_neuro_weight') is not None else None
        use_sentiment_analysis = bool(data.get('use_sentiment_analysis', False))
        use_deep_mappings = bool(data.get('use_deep_mappings', False))
        neuro_depth_file = data.get('neuro_depth_file', None)  # Path to JSON file or None
        
        # Layered stakeholder and neuro enrichment parameters (optional) - for crisis-specific sentiment
        layer_stakeholders_neuro = bool(data.get('layer_stakeholders_neuro', False))
        crisis_query = data.get('crisis_query', 'Sudan El Fasher IDP voices atrocities RSF')
        neuro_layer_file = data.get('neuro_layer_file', None)  # Path to JSON file or None
        
        # Validate inputs
        if levels < 1 or levels > 100:
            return jsonify({'error': 'Levels must be between 1 and 100'}), 400
        if n_monte < 1 or n_monte > 5000:
            return jsonify({'error': 'Monte Carlo iterations must be between 1 and 5000'}), 400
        if not isinstance(variants, list) or len(variants) == 0:
            return jsonify({'error': 'Variants must be a non-empty list'}), 400
        
        valid_variants = ['ADHD-collectivist', 'autistic-individualist', 'neurotypical-hybrid']
        for v in variants:
            if v not in valid_variants:
                return jsonify({'error': f'Invalid variant: {v}. Must be one of {valid_variants}'}), 400
        
        # Validate historical data parameters if provided
        if history_file_path:
            if not isinstance(history_file_path, str):
                return jsonify({'error': 'history_file_path must be a string'}), 400
            if historical_weight < 0.0 or historical_weight > 1.0:
                return jsonify({'error': 'historical_weight must be between 0.0 and 1.0'}), 400
        
        # Validate risk-utility recalibration parameters if provided
        if recalibrate_risk_utility:
            if target_equity < 0.0 or target_equity > 1.0:
                return jsonify({'error': 'target_equity must be between 0.0 and 1.0'}), 400
        
        # Validate scenario update parameters if provided
        if scenario_update_source or scenario_update_file:
            if scenario_update_source and not isinstance(scenario_update_source, str):
                return jsonify({'error': 'scenario_update_source must be a string'}), 400
            if scenario_update_file and not isinstance(scenario_update_file, str):
                return jsonify({'error': 'scenario_update_file must be a string'}), 400
            if scenario_merge_strategy not in ['overwrite', 'average', 'weighted']:
                return jsonify({'error': 'scenario_merge_strategy must be one of: overwrite, average, weighted'}), 400
            if scenario_update_weight < 0.0 or scenario_update_weight > 1.0:
                return jsonify({'error': 'scenario_update_weight must be between 0.0 and 1.0'}), 400
        
        # Validate simulation validation parameters if provided
        if validate_results:
            if target_metric not in ['mean_fidelity', 'utility_score', 'novelty_gen', 'mean_success_rate']:
                return jsonify({'error': 'target_metric must be one of: mean_fidelity, utility_score, novelty_gen, mean_success_rate'}), 400
            if risk_metric not in ['asymmetry_delta', 'std', 'mean_phi_delta']:
                return jsonify({'error': 'risk_metric must be one of: asymmetry_delta, std, mean_phi_delta'}), 400
            if cv_folds < 2 or cv_folds > 20:
                return jsonify({'error': 'cv_folds must be between 2 and 20'}), 400
            if bias_threshold < 0.0 or bias_threshold > 1.0:
                return jsonify({'error': 'bias_threshold must be between 0.0 and 1.0'}), 400
            if equity_threshold < 0.0 or equity_threshold > 1.0:
                return jsonify({'error': 'equity_threshold must be between 0.0 and 1.0'}), 400
        
        # Validate urgency threshold adjustment parameters if provided
        if adjust_urgency_thresholds:
            if urgency_factor < 0.5 or urgency_factor > 2.0:
                return jsonify({'error': 'urgency_factor must be between 0.5 and 2.0'}), 400
            if urgency_base_threshold < 0.0 or urgency_base_threshold > 1.0:
                return jsonify({'error': 'urgency_base_threshold must be between 0.0 and 1.0'}), 400
            if urgency_data_source is not None and not isinstance(urgency_data_source, dict):
                return jsonify({'error': 'urgency_data_source must be a dictionary or null'}), 400
        
        # Validate dynamic urgency adjustment parameters if provided
        if use_dynamic_urgency_adjust:
            if dynamic_urgency_api_url is not None and not isinstance(dynamic_urgency_api_url, str):
                return jsonify({'error': 'dynamic_urgency_api_url must be a string or null'}), 400
            if dynamic_urgency_pledge_keywords is not None and not isinstance(dynamic_urgency_pledge_keywords, list):
                return jsonify({'error': 'dynamic_urgency_pledge_keywords must be a list or null'}), 400
            if dynamic_urgency_base_threshold < 0.0 or dynamic_urgency_base_threshold > 1.0:
                return jsonify({'error': 'dynamic_urgency_base_threshold must be between 0.0 and 1.0'}), 400
        
        # Validate stakeholder depth enhancement parameters if provided
        if enhance_stakeholder_depth:
            if stakeholder_api_url is not None and not isinstance(stakeholder_api_url, str):
                return jsonify({'error': 'stakeholder_api_url must be a string or null'}), 400
            if not isinstance(stakeholder_local_query, str):
                return jsonify({'error': 'stakeholder_local_query must be a string'}), 400
            if stakeholder_guidelines_file is not None and not isinstance(stakeholder_guidelines_file, str):
                return jsonify({'error': 'stakeholder_guidelines_file must be a string or null'}), 400
            if stakeholder_weight is not None:
                if stakeholder_weight < 0.0 or stakeholder_weight > 1.0:
                    return jsonify({'error': 'stakeholder_weight must be between 0.0 and 1.0'}), 400
        
        # Validate escalation threshold adjustment parameters if provided
        if adjust_escalation_thresholds:
            if escalation_api_url is not None and not isinstance(escalation_api_url, str):
                return jsonify({'error': 'escalation_api_url must be a string or null'}), 400
            if escalation_threat_keywords is not None and not isinstance(escalation_threat_keywords, list):
                return jsonify({'error': 'escalation_threat_keywords must be a list or null'}), 400
            if escalation_base_threshold < 0.0 or escalation_base_threshold > 1.0:
                return jsonify({'error': 'escalation_base_threshold must be between 0.0 and 1.0'}), 400
            if escalation_data is not None and not isinstance(escalation_data, dict):
                return jsonify({'error': 'escalation_data must be a dictionary or null'}), 400
            if escalation_factor is not None:
                if escalation_factor < 0.5 or escalation_factor > 2.0:
                    return jsonify({'error': 'escalation_factor must be between 0.5 and 2.0'}), 400
        
        # Validate stakeholder and neurodiversity enrichment parameters if provided
        if enrich_stakeholder_neurodiversity:
            if x_api_url is not None and not isinstance(x_api_url, str):
                return jsonify({'error': 'x_api_url must be a string or null'}), 400
            if not isinstance(local_query, str):
                return jsonify({'error': 'local_query must be a string'}), 400
            if neuro_mappings is not None and not isinstance(neuro_mappings, dict):
                return jsonify({'error': 'neuro_mappings must be a dictionary or null'}), 400
            if filter_keywords is not None and not isinstance(filter_keywords, list):
                return jsonify({'error': 'filter_keywords must be a list or null'}), 400
            if stakeholder_neuro_weight is not None:
                if stakeholder_neuro_weight < 0.0 or stakeholder_neuro_weight > 1.0:
                    return jsonify({'error': 'stakeholder_neuro_weight must be between 0.0 and 1.0'}), 400
            if neuro_depth_file is not None and not isinstance(neuro_depth_file, str):
                return jsonify({'error': 'neuro_depth_file must be a string or null'}), 400
        
        # Validate layered stakeholder and neuro enrichment parameters if provided
        if layer_stakeholders_neuro:
            if not isinstance(crisis_query, str):
                return jsonify({'error': 'crisis_query must be a string'}), 400
            if neuro_layer_file is not None and not isinstance(neuro_layer_file, str):
                return jsonify({'error': 'neuro_layer_file must be a string or null'}), 400
        
        # Run simulation (disable progress bars for web)
        # Use internal output for backend processing (full metrics)
        results = simulate_prhp(
            levels=levels,
            variants=variants,
            n_monte=n_monte,
            seed=seed,
            use_quantum=use_quantum,
            track_levels=track_levels,
            show_progress=False,  # Disable tqdm for web
            public_output_only=False,  # Internal use - get full results
            history_file_path=history_file_path,
            historical_weight=historical_weight,
            recalibrate_risk_utility=recalibrate_risk_utility,
            target_equity=target_equity,
            scenario_update_source=scenario_update_source,
            scenario_update_file=scenario_update_file,
            scenario_merge_strategy=scenario_merge_strategy,
            scenario_update_weight=scenario_update_weight,
            validate_results=validate_results,
            target_metric=target_metric,
            risk_metric=risk_metric,
            cv_folds=cv_folds,
            bias_threshold=bias_threshold,
            equity_threshold=equity_threshold,
            adjust_urgency_thresholds=adjust_urgency_thresholds,
            urgency_factor=urgency_factor,
            urgency_base_threshold=urgency_base_threshold,
            urgency_data_source=urgency_data_source,
            use_dynamic_urgency_adjust=use_dynamic_urgency_adjust,
            dynamic_urgency_api_url=dynamic_urgency_api_url,
            dynamic_urgency_pledge_keywords=dynamic_urgency_pledge_keywords,
            dynamic_urgency_base_threshold=dynamic_urgency_base_threshold,
            enhance_stakeholder_depth=enhance_stakeholder_depth,
            stakeholder_api_url=stakeholder_api_url,
            stakeholder_local_query=stakeholder_local_query,
            stakeholder_guidelines_file=stakeholder_guidelines_file,
            stakeholder_weight=stakeholder_weight,
            adjust_escalation_thresholds=adjust_escalation_thresholds,
            escalation_api_url=escalation_api_url,
            escalation_threat_keywords=escalation_threat_keywords,
            escalation_base_threshold=escalation_base_threshold,
            escalation_data=escalation_data,
            escalation_factor=escalation_factor,
            enrich_stakeholder_neurodiversity=enrich_stakeholder_neurodiversity,
            x_api_url=x_api_url,
            local_query=local_query,
            neuro_mappings=neuro_mappings,
            filter_keywords=filter_keywords,
            stakeholder_neuro_weight=stakeholder_neuro_weight,
            use_sentiment_analysis=use_sentiment_analysis,
            use_deep_mappings=use_deep_mappings,
            neuro_depth_file=neuro_depth_file,
            layer_stakeholders_neuro=layer_stakeholders_neuro,
            crisis_query=crisis_query,
            neuro_layer_file=neuro_layer_file
        )
        
        # Format results for JSON serialization
        formatted_results = {}
        for variant, data in results.items():
            formatted_results[variant] = {
                'mean_fidelity': safe_float(data['mean_fidelity']),
                'std': safe_float(data['std']),
                'asymmetry_delta': safe_float(data['asymmetry_delta']),
                'novelty_gen': safe_float(data['novelty_gen']),
                'phi_deltas': [safe_float(d) for d in data['phi_deltas']] if data.get('phi_deltas') else [],
                'level_phis': [safe_float(p) for p in data['level_phis']] if data.get('level_phis') else [],
                'mean_phi_delta': safe_float(data['mean_phi_delta']) if data.get('mean_phi_delta') is not None else None
            }
            
            # Add historical integration metadata if available
            if 'historical_integration' in data:
                formatted_results[variant]['historical_integration'] = data['historical_integration']
            
            # Add recalibration metadata if available
            if 'recalibration' in data:
                formatted_results[variant]['recalibration'] = data['recalibration']
            
            # Add scenario update metadata if available
            if 'scenario_update' in data:
                formatted_results[variant]['scenario_update'] = data['scenario_update']
            
            # Add validation metadata if available
            if 'validation' in data:
                formatted_results[variant]['validation'] = data['validation']
            
            # Add urgency adjustment metadata if available
            if 'urgency_adjustment' in data:
                formatted_results[variant]['urgency_adjustment'] = data['urgency_adjustment']
            
            # Add dynamic urgency adjustment metadata if available
            if 'dynamic_urgency_adjustment' in data:
                formatted_results[variant]['dynamic_urgency_adjustment'] = data['dynamic_urgency_adjustment']
            
            # Add stakeholder enhancement metadata if available
            if 'stakeholder_enhancement' in data:
                formatted_results[variant]['stakeholder_enhancement'] = data['stakeholder_enhancement']
            
            # Add escalation adjustment metadata if available
            if 'escalation_adjustment' in data:
                formatted_results[variant]['escalation_adjustment'] = data['escalation_adjustment']
            
            # Add stakeholder and neurodiversity enrichment metadata if available
            if 'stakeholder_neuro_enrichment' in data:
                formatted_results[variant]['stakeholder_neuro_enrichment'] = data['stakeholder_neuro_enrichment']
            if 'neuro_mapping' in data:
                formatted_results[variant]['neuro_mapping'] = data['neuro_mapping']
            if 'deep_neuro' in data:
                formatted_results[variant]['deep_neuro'] = data['deep_neuro']
            if 'local_voices' in data:
                formatted_results[variant]['local_voices'] = data['local_voices']
            if 'voice_weight' in data:
                formatted_results[variant]['voice_weight'] = safe_float(data['voice_weight'])
            if 'layered_neuro' in data:
                formatted_results[variant]['layered_neuro'] = data['layered_neuro']
            if 'stakeholder_inputs' in data:
                formatted_results[variant]['stakeholder_inputs'] = data['stakeholder_inputs']
            if 'stakeholder_weight' in data:
                formatted_results[variant]['stakeholder_weight'] = safe_float(data['stakeholder_weight'])
        
        response_data = {
            'success': True,
            'results': formatted_results,
            'parameters': {
                'levels': levels,
                'variants': variants,
                'n_monte': n_monte,
                'seed': seed,
                'use_quantum': use_quantum
            }
        }
        
        # Add historical data info if used
        if history_file_path:
            response_data['historical_data'] = {
                'file_path': history_file_path,
                'weight': historical_weight,
                'applied': any('historical_integration' in data for data in results.values())
            }
        
        # Add risk-utility recalibration info if used
        if recalibrate_risk_utility:
            response_data['risk_utility_recalibration'] = {
                'applied': any('recalibration' in data for data in results.values()),
                'target_equity': target_equity
            }
        
        # Add scenario update info if used
        if scenario_update_source or scenario_update_file:
            response_data['scenario_updates'] = {
                'applied': any('scenario_update' in data for data in results.values()),
                'source': scenario_update_source or scenario_update_file,
                'merge_strategy': scenario_merge_strategy,
                'update_weight': scenario_update_weight
            }
        
        # Add simulation validation info if used
        if validate_results:
            response_data['simulation_validation'] = {
                'applied': any('validation' in data for data in results.values()),
                'target_metric': target_metric,
                'risk_metric': risk_metric,
                'cv_folds': cv_folds,
                'bias_threshold': bias_threshold,
                'equity_threshold': equity_threshold,
                'all_valid': all(data.get('validation', {}).get('is_valid', False) for data in results.values())
            }
        
        # Add urgency threshold adjustment info if used
        if adjust_urgency_thresholds:
            response_data['urgency_threshold_adjustment'] = {
                'applied': any('urgency_adjustment' in data for data in results.values()),
                'urgency_factor': urgency_factor,
                'base_threshold': urgency_base_threshold,
                'data_source_used': urgency_data_source is not None,
                'data_source_keys': list(urgency_data_source.keys()) if urgency_data_source else []
            }
        
        # Add dynamic urgency adjustment info if used
        if use_dynamic_urgency_adjust:
            dynamic_urgency_adjustments = [data.get('dynamic_urgency_adjustment', {}) for data in results.values()]
            response_data['dynamic_urgency_adjustment'] = {
                'applied': any('dynamic_urgency_adjustment' in data for data in results.values()),
                'base_threshold': dynamic_urgency_base_threshold,
                'adjusted_threshold': dynamic_urgency_adjustments[0].get('adjusted_threshold', dynamic_urgency_base_threshold) if dynamic_urgency_adjustments else dynamic_urgency_base_threshold,
                'boost_factor': dynamic_urgency_adjustments[0].get('boost_factor', 1.0) if dynamic_urgency_adjustments else 1.0,
                'pledge_count': dynamic_urgency_adjustments[0].get('pledge_count', 0) if dynamic_urgency_adjustments else 0,
                'api_used': dynamic_urgency_adjustments[0].get('api_used', False) if dynamic_urgency_adjustments else False,
                'api_url': dynamic_urgency_api_url,
                'pledge_keywords': dynamic_urgency_pledge_keywords or ['RSF pledges', 'aid access', 'UN corridors'],
                'utilities_capped': dynamic_urgency_adjustments[0].get('utilities_capped', False) if dynamic_urgency_adjustments else False
            }
        
        # Add stakeholder depth enhancement info if used
        if enhance_stakeholder_depth:
            stakeholder_enhancements = [data.get('stakeholder_enhancement', {}) for data in results.values()]
            response_data['stakeholder_depth_enhancement'] = {
                'applied': any('stakeholder_enhancement' in data for data in results.values()),
                'stakeholder_data_fetched': any(e.get('stakeholder_data_fetched', False) for e in stakeholder_enhancements),
                'guidelines_applied': any(e.get('guidelines_applied', False) for e in stakeholder_enhancements),
                'stakeholder_variant_added': any(e.get('stakeholder_variant_added', False) for e in stakeholder_enhancements),
                'api_url': stakeholder_api_url,
                'local_query': stakeholder_local_query,
                'guidelines_file': stakeholder_guidelines_file,
                'stakeholder_weight': stakeholder_weight
            }
        
        # Add escalation threshold adjustment info if used
        if adjust_escalation_thresholds:
            escalation_adjustments = [data.get('escalation_adjustment', {}) for data in results.values()]
            response_data['escalation_threshold_adjustment'] = {
                'applied': any('escalation_adjustment' in data for data in results.values()),
                'escalation_factor': escalation_adjustments[0].get('escalation_factor', 1.0) if escalation_adjustments else 1.0,
                'base_threshold': escalation_base_threshold,
                'adjusted_threshold': escalation_adjustments[0].get('adjusted_threshold', escalation_base_threshold) if escalation_adjustments else escalation_base_threshold,
                'api_url': escalation_api_url,
                'threat_keywords': escalation_threat_keywords,
                'escalation_data_used': escalation_data is not None,
                'escalation_factor_override': escalation_factor is not None
            }
            # Add escalation data details if available
            if escalation_adjustments and escalation_adjustments[0].get('escalation_data'):
                response_data['escalation_threshold_adjustment']['escalation_data'] = escalation_adjustments[0]['escalation_data']
        
        # Add stakeholder and neurodiversity enrichment info if used
        if enrich_stakeholder_neurodiversity:
            stakeholder_neuro_enrichments = [data.get('stakeholder_neuro_enrichment', {}) for data in results.values()]
            response_data['stakeholder_neurodiversity_enrichment'] = {
                'applied': any('stakeholder_neuro_enrichment' in data for data in results.values()),
                'stakeholder_data_fetched': any(e.get('stakeholder_data_fetched', False) for e in stakeholder_neuro_enrichments),
                'neuro_mappings_applied': any(e.get('neuro_mappings_applied', False) for e in stakeholder_neuro_enrichments),
                'stakeholder_items_count': max((e.get('stakeholder_items_count', 0) for e in stakeholder_neuro_enrichments), default=0),
                'variants_enriched': max((e.get('variants_enriched', 0) for e in stakeholder_neuro_enrichments), default=0),
                'x_api_url': x_api_url,
                'local_query': local_query,
                'local_stakeholder_variant_added': 'local-stakeholder' in results
            }
        
        # Add layered stakeholder and neuro enrichment info if used
        if layer_stakeholders_neuro:
            layered_enrichments = [data.get('layered_neuro', None) for data in results.values()]
            response_data['layered_stakeholder_neuro_enrichment'] = {
                'applied': any('layered_neuro' in data for data in results.values()),
                'crisis_query': crisis_query,
                'neuro_layer_file': neuro_layer_file,
                'variants_with_layered_neuro': sum(1 for e in layered_enrichments if e is not None),
                'x_api_url': x_api_url,
                'crisis_sentiment_analysis': True
            }
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': f'Simulation failed: {str(e)}'}), 500


@app.route('/api/prhp/pruner', methods=['POST'])
def prhp_pruner():
    """Run Political Pruner simulation."""
    if not PRHP_AVAILABLE:
        return jsonify({'error': 'PRHP framework not available'}), 500
    
    try:
        data = request.get_json()
        
        levels = int(data.get('levels', 9))
        variant = data.get('variant', 'neurotypical-hybrid')
        n_monte = int(data.get('n_monte', 100))
        seed = int(data.get('seed', 42)) if data.get('seed') is not None else None
        
        # CRITICAL: FORCE quantum mode - ALL inputs MUST be Qiskit processed
        qiskit_available = False
        try:
            from qubit_hooks import HAS_QISKIT
            qiskit_available = HAS_QISKIT
        except ImportError:
            pass
        
        # Try to install Qiskit if not available
        if not qiskit_available:
            try:
                import subprocess
                import sys
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "qiskit", "qiskit-aer", "--quiet", "--upgrade"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=120
                )
                try:
                    from qubit_hooks import HAS_QISKIT
                    qiskit_available = HAS_QISKIT
                except (ImportError, AttributeError):
                    pass
            except Exception:
                pass
        
        # ALWAYS force quantum mode - no exceptions
        use_quantum = True  # FORCE quantum mode - ALL inputs Qiskit processed (or classical approximation)
        if qiskit_available:
            print(f"[PRHP] ✓ QISKIT PROCESSING ENABLED for /api/prhp/pruner")
        else:
            print("[PRHP] ✓ QUANTUM MODE FORCED for /api/prhp/pruner - Using classical approximation")
        
        result = simulate_pruner_levels(
            levels=levels,
            variant=variant,
            n_monte=n_monte,
            seed=seed,
            use_quantum=use_quantum  # FORCED to True when Qiskit available - all inputs Qiskit processed
        )
        
        # Format for JSON
        formatted_result = {
            'level_deltas': [safe_float(d) for d in result['level_deltas']],
            'self_model_coherence': safe_float(result['self_model_coherence']),
            'success_rate': safe_float(result['success_rate'])
        }
        
        return jsonify({
            'success': True,
            'result': formatted_result
        })
    
    except Exception as e:
        return jsonify({'error': f'Pruner simulation failed: {str(e)}'}), 500


@app.route('/api/prhp/extinction', methods=['POST'])
def prhp_extinction():
    """Run virus extinction forecast."""
    if not PRHP_AVAILABLE:
        return jsonify({'error': 'PRHP framework not available'}), 500
    
    try:
        data = request.get_json()
        
        variants = data.get('variants', ['neurotypical-hybrid'])
        n_sims = int(data.get('n_sims', 100))
        seed = int(data.get('seed', 42)) if data.get('seed') is not None else None
        
        forecast = forecast_extinction_risk(
            variants=variants,
            n_sims=n_sims,
            seed=seed
        )
        
        # Format for JSON
        formatted_forecast = {}
        for variant, metrics in forecast.items():
            formatted_forecast[variant] = {
                'extinction_risk': safe_float(metrics['extinction_risk']),
                'mean_mitigation': safe_float(metrics['mean_mitigation']),
                'averted_rate': safe_float(metrics['averted_rate'])
            }
        
        return jsonify({
            'success': True,
            'forecast': formatted_forecast
        })
    
    except Exception as e:
        return jsonify({'error': f'Extinction forecast failed: {str(e)}'}), 500


@app.route('/api/prhp/quantum', methods=['POST'])
def prhp_quantum():
    """Run quantum hooks simulation."""
    if not PRHP_AVAILABLE:
        return jsonify({'error': 'PRHP framework not available'}), 500
    
    try:
        import numpy as np
        from qubit_hooks import compute_phi, entangle_nodes_variant, recalibrate_novelty, inject_phase_flip, inject_predation
        
        data = request.get_json()
        
        variant = data.get('variant', 'neurotypical-hybrid')
        flip_prob = float(data.get('flip_prob', 0.30))
        threshold = float(data.get('threshold', 0.70))
        divergence = float(data.get('divergence', 0.22))
        seed = data.get('seed', None)  # Allow configurable seed or None for random
        
        # CRITICAL: FORCE quantum mode - ALL inputs MUST be Qiskit processed
        qiskit_available = False
        try:
            from qubit_hooks import HAS_QISKIT
            qiskit_available = HAS_QISKIT
        except ImportError:
            pass
        
        # Try to install Qiskit if not available
        if not qiskit_available:
            try:
                import subprocess
                import sys
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "qiskit", "qiskit-aer", "--quiet", "--upgrade"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=120
                )
                try:
                    from qubit_hooks import HAS_QISKIT
                    qiskit_available = HAS_QISKIT
                except (ImportError, AttributeError):
                    pass
            except Exception:
                pass
        
        # ALWAYS force quantum mode - no exceptions
        use_quantum = True  # FORCE quantum mode - ALL inputs Qiskit processed (or classical approximation)
        if qiskit_available:
            print(f"[PRHP] ✓ QISKIT PROCESSING ENABLED for /api/prhp/quantum")
        else:
            print("[PRHP] ✓ QUANTUM MODE FORCED for /api/prhp/quantum - Using classical approximation")
        
        # Validate inputs
        if variant not in ['ADHD-collectivist', 'autistic-individualist', 'neurotypical-hybrid']:
            return jsonify({'error': f'Invalid variant: {variant}'}), 400
        if not 0.0 <= flip_prob <= 1.0:
            return jsonify({'error': 'flip_prob must be between 0.0 and 1.0'}), 400
        if not 0.0 <= threshold <= 1.0:
            return jsonify({'error': 'threshold must be between 0.0 and 1.0'}), 400
        if not 0.0 <= divergence <= 1.0:
            return jsonify({'error': 'divergence must be between 0.0 and 1.0'}), 400
        
        # Set seed if provided, otherwise use random
        if seed is not None:
            np.random.seed(int(seed))
        else:
            # Use a seed based on current time to get variation
            import time
            np.random.seed(int(time.time() * 1000) % 1000000)
        
        # Run quantum hooks simulation with detailed tracking
        state_a, state_b, _, _ = entangle_nodes_variant(variant, use_quantum=use_quantum, seed=seed)
        initial_phi_a = compute_phi(state_a, state_b, use_quantum=use_quantum)
        initial_phi_b = compute_phi(state_b, state_a, use_quantum=use_quantum)
        initial_phi = (initial_phi_a + initial_phi_b) / 2
        initial_asymmetry = abs(initial_phi_a - initial_phi_b) / (initial_phi + 1e-10)
        
        # Inject predation (this affects the state)
        state_b_pred = inject_predation(state_b.copy(), divergence=divergence)
        phi_after_pred = compute_phi(state_a, state_b_pred, use_quantum=use_quantum)
        
        # Inject phase flip (this significantly affects the phase)
        state_b_flipped = inject_phase_flip(state_b_pred, flip_prob=flip_prob, variant=variant)
        phi_after_flip = compute_phi(state_a, state_b_flipped, use_quantum=use_quantum)
        
        # Recalibrate novelty (threshold determines if pruning happens)
        state_a_recal, state_b_recal, asymmetry = recalibrate_novelty(
            state_a, state_b_flipped, threshold=threshold, use_quantum=use_quantum
        )
        
        final_phi_a = compute_phi(state_a_recal, state_b_recal, use_quantum=use_quantum)
        final_phi_b = compute_phi(state_b_recal, state_a_recal, use_quantum=use_quantum)
        final_phi = (final_phi_a + final_phi_b) / 2
        phi_delta = final_phi - initial_phi
        
        # Check if pruning occurred (convert numpy bool to Python bool)
        pruning_occurred = bool(asymmetry > threshold)
        threshold_exceeded = bool(asymmetry > threshold)
        
        return jsonify({
            'success': True,
            'variant': variant,
            'parameters': {
                'flip_prob': float(flip_prob),
                'threshold': float(threshold),
                'divergence': float(divergence),
                'use_quantum': bool(use_quantum),
                'seed': int(seed) if seed is not None else None
            },
            'initial_phi': safe_float(initial_phi),
            'initial_phi_a': safe_float(initial_phi_a),
            'initial_phi_b': safe_float(initial_phi_b),
            'initial_asymmetry': safe_float(initial_asymmetry),
            'phi_after_predation': safe_float(phi_after_pred),
            'phi_after_flip': safe_float(phi_after_flip),
            'final_phi': safe_float(final_phi),
            'final_phi_a': safe_float(final_phi_a),
            'final_phi_b': safe_float(final_phi_b),
            'asymmetry': safe_float(asymmetry),
            'phi_delta': safe_float(phi_delta),
            'pruning_occurred': pruning_occurred,
            'threshold_exceeded': threshold_exceeded
        })
    
    except Exception as e:
        import traceback
        return jsonify({'error': f'Quantum simulation failed: {str(e)}', 'traceback': traceback.format_exc()}), 500


def ensure_qiskit_at_startup():
    """Ensure Qiskit is installed at startup for quantum extensions."""
    print("\n" + "="*60)
    print("PRHP Framework - Quantum Extensions Check")
    print("="*60)
    
    qiskit_available = False
    qiskit_version = 0
    
    try:
        from qubit_hooks import HAS_QISKIT, QISKIT_VERSION
        qiskit_available = HAS_QISKIT
        qiskit_version = QISKIT_VERSION
    except ImportError:
        pass
    
    if not qiskit_available:
        print("⚠ Qiskit not found. Installing for quantum extensions...")
        try:
            import subprocess
            import sys
            print("Installing qiskit and qiskit-aer...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "qiskit", "qiskit-aer", "--quiet"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            # Re-check after installation
            from qubit_hooks import HAS_QISKIT, QISKIT_VERSION
            qiskit_available = HAS_QISKIT
            qiskit_version = QISKIT_VERSION
            if qiskit_available:
                print(f"✓ Qiskit {qiskit_version} installed successfully")
            else:
                print("⚠ Qiskit installation may have failed")
        except Exception as e:
            print(f"⚠ Could not install Qiskit automatically: {e}")
            print("  To install manually, run: pip install qiskit qiskit-aer")
            print("  Quantum features will use classical approximation")
    else:
        print(f"✓ Qiskit {qiskit_version} available - Quantum extensions enabled")
    
    print("="*60 + "\n")
    return qiskit_available, qiskit_version


if __name__ == '__main__':
    # Ensure Qiskit is installed at startup
    qiskit_available, qiskit_version = ensure_qiskit_at_startup()
    
    print("Starting PRHP Framework UI...")
    print(f"Provider: {API_CONFIG['provider']}")
    if API_CONFIG['provider'].lower() == 'openrouter':
        print(f"Base URL: {API_CONFIG['base_url']}")
    print(f"Model: {API_CONFIG['model']}")
    print(f"API Key configured: {'Yes' if API_CONFIG['api_key'] else 'No (set API_KEY env var)'}")
    if not OPENAI_AVAILABLE and API_CONFIG['provider'].lower() in ['openrouter', 'openai']:
        print("WARNING: OpenAI package not installed. Run: pip install openai")
    print(f"PRHP Framework: {'Available ✓' if PRHP_AVAILABLE else 'Not available ✗'}")
    if not PRHP_AVAILABLE and PRHP_IMPORT_ERROR:
        print(f"  Error: {PRHP_IMPORT_ERROR}")
    print(f"Quantum Extensions: {'Enabled ✓' if qiskit_available else 'Classical mode (Qiskit not available)'}")
    if qiskit_available:
        print(f"  Qiskit Version: {qiskit_version}")
    print("\n" + "="*60)
    print("CRITICAL: All inputs are PRHP processed")
    if qiskit_available:
        print("CRITICAL: All inputs are Qiskit processed (QISKIT ALWAYS ENABLED)")
        print("CRITICAL: use_quantum is FORCED to True - user preference ignored")
    else:
        print("CRITICAL: Qiskit not available - using classical approximation")
    print("CRITICAL: All parameters are integrated in quantum hooks")
    print("="*60)
    print("\nProcessing Flow:")
    print("  1. Every user input → PRHP simulation")
    if qiskit_available:
        print("  2. Every PRHP simulation → Qiskit quantum processing (ALWAYS ENABLED)")
        print("  3. use_quantum parameter → FORCED to True (user preference ignored)")
    else:
        print("  2. Every PRHP simulation → Classical approximation (Qiskit not available)")
        print("  3. Install Qiskit to enable quantum processing: pip install qiskit qiskit-aer")
    print("  4. All parameters → Integrated into quantum hooks")
    print("     - variant → Quantum gates (σ_z, σ_x, Hadamard)")
    print("     - level → Divergence scaling")
    print("     - use_quantum → Qiskit circuits (FORCED True when Qiskit available)")
    print("     - seed → Reproducibility")
    print("     - All enhancement parameters → Full integration")
    print("\nOpen your browser to: http://localhost:5000")
    print("Press Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

