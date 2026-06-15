
import math
import numpy as np
import pandas as pd
import re

GLOBAL_COUNTRY_MAPPING = {
    'UK': 'UNITED KINGDOM',
    'SCOTLAND, UK': 'UNITED KINGDOM',
    'SCOTLAND': 'UNITED KINGDOM',
    'UNITED KINGDOM': 'UNITED KINGDOM',
    'USA': 'UNITED STATES',
    'UNITED STATES': 'UNITED STATES',
    'HOLLAND': 'NETHERLANDS',
    'NETHERLANDS': 'NETHERLANDS',
    'THE NETHERLANDS': 'NETHERLANDS',
    'SOUTH KOREA': 'REPUBLIC OF KOREA',
    'KOREA': 'REPUBLIC OF KOREA',
    'REPUBLIC OF KOREA': 'REPUBLIC OF KOREA',
    'EUROPE (FRANCE)': 'FRANCE',
}

_NR_TOKENS = {'NR', 'NOT REPORTED', 'NOT SPECIFIED', 'NAN', 'NONE', '', 'N/A', 'NA'}

def _is_nr(val) -> bool:
    """Return True if val is any flavour of 'not reported'."""
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    return str(val).strip().upper() in _NR_TOKENS

def extract_decision_year(val) -> str:
    """Extract the 4-digit calendar year from MM/DD/YYYY date strings (field A3).

    Args:
        val: A string, datetime object, or numeric value representing a date (A3 field).

    Returns:
        4-digit year string, or 'NR'.
    """
    if _is_nr(val):
        return 'NR'
    val_str = str(val).strip()
    if '/' in val_str:
        return val_str.split('/')[-1].strip()
    return 'NR'


def extract_clearance_pathway(pathway) -> str:
    """Isolate the first token of the Clearance_Info semicolon string (field A4).

    Format: 'Pathway; Product Code; Regulation Number'

    Args:
        pathway: Raw string or object representing the FDA submission type (A4 field).

    Returns:
        A standardized string category: '510(k)', 'De Novo', 'PMA', or 'Unknown'.
    """
    if _is_nr(pathway):
        return 'NR'
    pathway_str = str(pathway).strip()
    if ';' in pathway_str:
        return pathway_str.split(';')[0].strip().upper()
    return pathway_str.upper()

def extract_nb_manufacturers(val) -> float:
    """Extract the count of scanner manufacturers from field F1.

    F1 format: 'Yes; GE Healthcare, Siemens, Philips, Canon; 4 manufacturers'
    Falls back to counting comma-separated names in segment 2.

    Args:
        val: Raw string or object representing the manufacturers of the 
            scanner (F1 field).

    Returns:
        float (np.nan if not parseable).
    """
    if _is_nr(val):
        return np.nan
    val_str = str(val).strip()

    if ';' in val_str:
        try:
            last_segment = val_str.split(';')[-1].strip()

            match = re.search(r'\d+', last_segment)
            if match:
                return float(match.group(0))
        except Exception:
            pass
            
    return np.nan


def extract_manufacturers_list(val) -> list:
    """ Extract the list of scanner manufacturer names (upper-cased) from field F1.

    Extract and standardize scanner manufacturers into exactly 5 categories:
    ['GE', 'SIEMENS', 'PHILIPS', 'CANON', 'OTHER'].

    Args:
        val: Raw string or object representing the manufacturers of the 
            scanner (F1 field).

    Returns:
        list of the manufacturers.
    """
    if _is_nr(val) or pd.isna(val):
        return []
        
    val_str = str(val).strip()
    segments = val_str.split(';')
    
    if len(segments) >= 2:
        manufacturers_part = segments[1].upper()

        manufacturers_part = re.sub(r'\b(CO\.?|LTD\.?|INC\.?|LLC\.?|GMBH|S\.P\.A\.|CORP\.?|CORPORATION|USA|ONLY|HEALTHCARE|MEDICAL SYSTEMS|MEDICAL SYSTEM|MEDICAL|ELECTRONICS|HEALTH|DIGITALDIAGNOST|ULTRASOUND)\b', '', manufacturers_part)

        raw_names = [m.strip() for m in manufacturers_part.split(',') if m.strip()]
        
        clean_list = []
        for name in raw_names:
            name_clean = re.sub(r'\(.*?\)', '', name) 
            name_clean = re.sub(r'[^A-Z0-9\s/+-]', '', name_clean)
            name_clean = name_clean.strip()
            
            garbage_tokens = [
                'BUT IMPLIED TO BE INCLUDED IN THE TABLE ON PAGE 8',
                'NOT EXPLICITLY LISTED IN THE PROVIDED TEXT',
                'NOT REPORTED', 'NOT SPECIFIED', 'NOT SPECIFIED IN THE TEXT',
                'UNKNOWN', 'UNKNOWN MANUFACTURER', 'OTHERS', 'OTHER'
            ]
            if name_clean in garbage_tokens or len(name_clean) <= 1:
                continue
                
            if any(k in name_clean for k in ['GE', 'GEHC', 'GENERAL ELECTRIC']):
                final_category = 'GE'
            elif any(k in name_clean for k in ['SIEMENS', 'SIMENS', 'HEALTHINEERS']):
                final_category = 'SIEMENS'
            elif any(k in name_clean for k in ['PHILIPS', 'PHILLIPS']):
                final_category = 'PHILIPS'
            elif any(k in name_clean for k in ['CANON', 'CANNON', 'TOSHIBA']):
                final_category = 'CANON'
            else:
                final_category = 'OTHER'

            if final_category not in clean_list:
                clean_list.append(final_category)
                
        return clean_list
        
    return []

def extract_multi_protocol(val) -> bool:
    """Return True if F3 reports multi-protocol testing ('Multi-protocol: Yes').

    F3 field format: Parameters: [detail]; Multi-protocol: [Yes/No/NR]. 

    Args: 
        val: Raw string of object representing the acquisition and protocol
            (F3 field).

    Returns:
        Bool : True if Multiprotocol is Yes. 
    """
    val_str = str(val).strip()
    if 'Multi-protocol:' in val_str:
        try:
            after_keyword = val_str.split('Multi-protocol:')[-1].strip().upper()
            first_word = after_keyword.replace('.', '').replace(';', '').split()[0]
            return first_word == 'YES'
        except Exception:
            pass
            
    return False


def extract_param_disclosure(val) -> bool:
    """Return True if F3 reports any acquisition parameters.

    F3 field format: Parameters: [detail]; Multi-protocol: [Yes/No/NR]. 
    Possible parameters: kVp, slice, kernel, field strength, etc.

    Agrs: 
        val: Raw string or object representing the acquisition and protocol
            (F3 field).

    Returns:
        Bool: True if F3 reports any acquisition parameters.
    """
    val_str = str(val).strip()
    if 'Parameters:' in val_str:
        try:
            param_part = val_str.split('Parameters:')[1].split(';')[0].strip().upper()
  
            if not param_part or len(param_part) <= 2:
                return False
                
            nr_keywords = [
                'NOT REPORTED', 
                'NOT SPECIFIED', 
                'NOT EXPLICITLY', 
                'UNKNOWN', 
                'NR', 'NA', 'NAN'
            ]
            if any(keyword in param_part for keyword in nr_keywords):
                return False
            return True
        
        except Exception:
            pass
            
    return False

def extract_j3_score(val) -> float:
    """Safely parse the Scanner_Diversity_Score (J3) to a numeric float."""
    try:
        return float(str(val).strip())
    except Exception:
        return np.nan
    
def clean_comparator_method(val) -> str:
    """Standardize the comparator method from field G1 using hierarchical priority.

    Resolves overlapping text definitions by applying a strict clinical hierarchy 
    from highest level of evidence (Ground Truth) to lowest (Predicate/Standalone),
    ensuring mutually exclusive categories for single-choice visualizations.

    Args:
        val: Raw string or object representing the evaluation comparator method 
            (G1 field).

    Returns:
        str: One of the standardized categories: 'Ground truth', 'Radiologist', 
             'Predicate', 'Standalone', or 'None/NR'.
    """
    if pd.isna(val):
        return 'None/NR'
        
    val_str = str(val).lower().strip()
    
    if _is_nr(val_str) or 'no comparator' in val_str:
        return 'None/NR'
        
    if 'ground truth' in val_str:
        return 'Ground truth'
        
    if 'radiologist' in val_str or 'reader' in val_str:
        return 'Radiologist'

    if 'predicate' in val_str or 'reference' in val_str:
        return 'Predicate'

    if 'standalone' in val_str:
        return 'Standalone'
        
    return 'None/NR'

def extract_sample_size(val) -> float:
    """Extract total validation sample size from field D1.

    The function isolates the text after the semicolon, ignores any numbers 
    inside parentheses (which usually represent sub-cohorts like positive/negative 
    counts), and sums all remaining main numeric values to find the total 
    sample size.

    Examples:
        - 'Retrospective; 1,234 studies (652 positive, 582 negative)' -> 1234.0
        - 'Both; 877 exams (retrospective), 200 exams (prospective)' -> 1077.0
        - 'Prospective; 60 (Shoulder), 60 (Knee), 65 (Hip)' -> 185.0
        - 'Simulated/Enriched; at least 11,000 DRRs' -> 11000.0
        - 'Retrospective; NR' -> NaN

    Args:
        val: Raw string or object representing the study design and size
            (D1 field).

    Returns:
        float: The sum of all main numeric values found after the first semicolon, 
               or np.nan if no valid number or 'NR' is found.
    """
    if _is_nr(val):
        return np.nan
        
    val_str = str(val).strip()
    
    if ';' in val_str:
        size_part = val_str.split(';')[1].strip()
        size_part_cleaned = re.sub(r'\(.*?\)', '', size_part)
        numbers = re.findall(r'[\d,]+', size_part_cleaned)

        if numbers:
            total = 0.0
            for num in numbers:
                clean_num = num.replace(',', '')
                if clean_num.isdigit():
                    total += float(clean_num)
            return total if total > 0 else np.nan
            
    return np.nan

def extract_countries_c1d(val) -> list:
    """Parse training data countries from field C1d into a clean upper-case list.

    Strips trailing 'only' qualifiers.
    Examples: USA; Germany; Japan │ USA only │ Not specified. 

    Agrs:
        val: Raw string or object representing the training data countries (C1d field).

    Returns:
        list: list of the countries.
    """
    if _is_nr(val):
        return []
    val_str = str(val).strip()
    parts = [p.replace('only', '').strip().upper() for p in val_str.split(';') if p.strip()]

    clean_parts = []
    for p in parts:
        if p not in _NR_TOKENS:
            mapped = GLOBAL_COUNTRY_MAPPING.get(p, p)
            if mapped not in clean_parts:
                clean_parts.append(mapped)
    return clean_parts

def extract_countries_d3(val) -> list:
    """Parse validation-site countries from field D3.

    Format: Sites: [names]; Countries: [list]; Types: [classification].
    Example: Sites: MGH; Cleveland Clinic; Johns Hopkins; Countries: USA; USA; USA;
        Types: Academic; Academic; Academic 

    Args: 
        val: Raw string or object representing validation sites details
            (D3 field).

    Returns:
        list: upper-case list of country tokens.
    """
    if _is_nr(val):
        return []
    val_str = str(val).strip()

    regional_exclusions = {
        'ASIA', 'EUROPE', 'EUROPEAN', 'AMERICAS', 'NORTH AMERICA', 
        'SOUTH AMERICA', 'EU COUNTRIES', 'EU', 'OUS', 'OTHER COUNTRIES'
    }

    if 'Countries:' in val_str:
        try:
            countries_block = val_str.split('Countries:')[1]
            if 'Types:' in countries_block:
                countries_block = countries_block.split('Types:')[0]
            
            raw_tokens = re.split(r'[;,]', countries_block)
            
            clean_countries = []
            for t in raw_tokens:
                token_clean = t.strip().upper()
                
                if not token_clean or token_clean in ['NR', 'NAN'] or token_clean in regional_exclusions:
                    continue

                if '7 COUNTRIES' in token_clean:
                    continue

                token_mapped = GLOBAL_COUNTRY_MAPPING.get(token_clean, token_clean)
                
                if token_mapped not in clean_countries:
                    clean_countries.append(token_mapped)
                    
            return clean_countries
        except Exception:
            pass
            
    return []

# Alias kept so any legacy call to extract_validation_country() also works
extract_validation_country = extract_countries_d3

def is_public_dataset(val) -> bool:
    """Return True if field C1b mentions a public repository or known public dataset.

    Example: Mayo Clinic (Rochester, MN, USA); NIH ChestX-ray14 (public);
        Private multi-center US data

    Args: 
        val: Raw string or object representing the training data source (C1b field).

    Returns:
        Bool: True if one source is public.
    """
    if _is_nr(val):
        return False
        
    val_str = str(val).lower().strip()
    exclusion_keywords = [
        'not specified', 
        'unspecified', 
        'unknown', 
        'private/public', 
        'public/private'
    ]
    if any(excl in val_str for excl in exclusion_keywords):
        return False
    
    public_keywords = [
        'public', 'nih', 'mimic', 'tcia', 'oasis', 'open-source', 
        'open access', 'uk biobank', 'chestx-ray14', 'luna16', 
        'github', 'kaggle', 'repository', 'worldwide population'
    ]
    return any(kw in val_str for kw in public_keywords)

def extract_hq_country(val) -> str:
    """Extract the manufacturer's HQ country from field B4.

    B4 format: HQ: Country (City); Market: [description]. 
    Example: HQ: USA (San Francisco, CA); Market: Not restricted (global implied). 

    Args:
        val: Raw string or object representing the company country and market
            (B4 field).

    Returns:
        str: upper-case country token, or 'NR'.
    """
    if _is_nr(val):
        return 'NR'
    val_str = str(val).strip()
    if 'HQ:' in val_str:
        try:
            hq_part = val_str.split('HQ:')[1].split(';')[0].split('(')[0]
            raw_country = hq_part.strip().upper()
            
            if raw_country in _NR_TOKENS or not raw_country:
                return 'NR'
            
            return GLOBAL_COUNTRY_MAPPING.get(raw_country, raw_country)
        except Exception:
            pass
    return 'NR'

def extract_validation_types(val) -> str:
    """Extract site classification (Academic / Community / Mixed) from field D3.

    D3 Format: Sites: [names]; Countries: [list]; Types: [classification].
    Example: Sites: MGH; Cleveland Clinic; Johns Hopkins; Countries: USA; USA; USA;
        Types: Academic; Academic; Academic 

    Args:
        val: Raw string or object representing the validation sites details
            (D3 field).

    Returns:
        str:lower-case string, or 'nr'
    """
    if _is_nr(val):
        return 'nr'
    val_str = str(val).strip()

    if 'Types:' in val_str:
        try:
            types_block = val_str.split('Types:')[-1].strip().lower()
            
            import re
            tokens = [t.strip() for t in re.split(r'[;,]', types_block) if t.strip()]
            tokens = [t for t in tokens if t not in ['nr', 'nan', 'not reported']]
            
            if not tokens:
                return 'nr'
                
            unique_tokens = set(tokens)

            if any('mixed' in t for t in unique_tokens) or len(unique_tokens) > 1:
                return 'mixed'
                
            actual_type = unique_tokens.pop()

            if 'academic' in actual_type or 'university' in actual_type:
                return 'academic'

            community_keywords = [
                'community', 'private', 'facility', 'clinical', 'clinic', 
                'hospital', 'trauma center', 'commercial', 'medical device', 
                'laboratory', 'core lab'
            ]
            if any(kw in actual_type for kw in community_keywords):
                return 'community'
                
            if 'bench' in actual_type or 'simulated' in actual_type:
                return 'simulated/bench'
                
            return actual_type 
            
        except Exception:
            pass
            
    return 'nr'

def extract_predicate_year(val) -> float:
    """Extract predicate clearance year from field H1.

    H1 Format: Predicate #: [number]; Name: [name]; Year: [year].
    Example: Predicate #: K210926; Name: Aidoc BriefCase ICH v1.0; Year: 2021 

    Agrs:
        val: Raw string or object representing the predicate chain (H1 field).

    Returns:
        float: integer year, or np.nan.
    """
    if _is_nr(val):
        return np.nan
    val_str = str(val).strip()
    if 'Year:' in val_str:
        try:
            year_str = val_str.split('Year:')[-1].strip()
            year_digits = ''.join(c for c in year_str if c.isdigit())
            if len(year_digits) >= 4:
                return float(year_digits[:4])
        except Exception:
            pass
    return np.nan

def extract_predicate_number(val):
    """Isolate the raw FDA submission ID from field H1.

    H1 Format: Predicate #: [number]; Name: [name]; Year: [year].
    Example: Predicate #: K210926; Name: Aidoc BriefCase ICH v1.0; Year: 2021 

    Agrs:
        val: Raw string or object representing the predicate chain (H1 field).

    Returns:
        string like 'K210926', or None.
    """
    if _is_nr(val):
        return None
    val_str = str(val).strip()
    if 'Predicate #:' in val_str:
        try:
            return val_str.split('Predicate #:')[1].split(';')[0].strip()
        except Exception:
            pass
    return None

def find_ancestor_chain(predicate_map: dict, submission_number) -> list:
    """Recursively trace the chain of predicates back to the root ancestor.

    Args:
        predicate_map : dictionnary  {submission_id -> direct_predicate_id}
        submission_number : string-like  starting device number. 

    Returns:
        list: a list of ancestor submission IDs, oldest last.

    Note: Ceiling of 50 iterations guards against circular references.
    """
    chain = []
    current = str(submission_number).strip()
    for _ in range(50):
        parent = predicate_map.get(current)
        if not parent or str(parent).strip().upper() in _NR_TOKENS or parent == current:
            break
        chain.append(parent)
        current = str(parent).strip()
    return chain

def extract_h3_subflag(val, prefix: str) -> str:
    """Parse a named sub-field out of the semicolon-delimited Regulatory_Flags string (H3).

    H3 Format: Third party: [Y/N/NR]; Recall: [Y/N/Check database]; NCT: [number or N/A];
        Expedited: [Y/N/NR].
    H3 Examples: Third party: No; Recall: No; NCT: N/A; Expedited: Yes – Breakthrough 

    Args:
        val: Raw string or object representing the H3 Regulatory_Flags field.
        prefix: The target sub-field identifier to search for (e.g., 'Recall:', 'NCT:').

    Returns:
        str: The cleaned, uppercase value extracted for the specified prefix. 
            Returns 'NR' (Not Recorded) if the value is missing, the prefix is 
            not found, or an parsing error occurs.

    Examples:
        >>> extract_h3_subflag("Recall: No; NCT: N/A", "Recall:")
        'NO'
        >>> extract_h3_subflag("Third party: No; Expedited: Yes – Breakthrough", "Expedited:")
        'YES – BREAKTHROUGH'
        >>> extract_h3_subflag("Third party: Yes", "Third party:")
        'YES'
        >>> extract_h3_subflag("NCT: NCT01943916", "NCT:")
        'NCT01943916'
    """
    if _is_nr(val):
        return 'NR'
    val_str = str(val).strip()
    if prefix in val_str:
        try:
            for component in val_str.split(';'):
                if prefix in component:
                    return component.split(prefix)[1].strip().upper()
        except Exception:
            pass
    return 'NR'

def extract_j4_score(val) -> int:
    """Parse the numeric score out of the Overall_Generalizability_Risk (J4) field.

    J4 format: 'Score: 8; Risk: Moderate risk'

    Args:
        val: Raw string or object representing the J4 score (J4 field).

    Returns:
        int: integer 0-15, or np.nan.
    """
    try:
        val_str = str(val)
        if 'Score:' in val_str:
            return int(val_str.split('Score:')[1].split(';')[0].strip())
    except Exception:
        pass
    return np.nan
