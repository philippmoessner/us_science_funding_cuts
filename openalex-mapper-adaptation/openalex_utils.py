import numpy as np
from urllib.parse import urlparse, parse_qs
from pyalex import Works, Authors, Institutions
import pandas as pd
import ast, json

def openalex_url_to_pyalex_query(url):
    """
    Convert an OpenAlex search URL to a pyalex query.
    
    Args:
    url (str): The OpenAlex search URL.
    
    Returns:
    tuple: (Works object, dict of parameters)
    """
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    
    # Initialize the Works object
    query = Works()
    
    # Handle filters
    if 'filter' in query_params:
        filters = query_params['filter'][0].split(',')
        for f in filters:
            if ':' in f:
                key, value = f.split(':', 1)
                if key == 'default.search':
                    query = query.search(value)
                else:
                    query = query.filter(**{key: value})
    
    # Handle sort - Fixed to properly handle field:direction format
    if 'sort' in query_params:
        sort_params = query_params['sort'][0].split(',')
        for s in sort_params:
            if ':' in s:  # Handle field:direction format
                field, direction = s.split(':')
                query = query.sort(**{field: direction})
            elif s.startswith('-'):  # Handle -field format
                query = query.sort(**{s[1:]: 'desc'})
            else:  # Handle field format
                query = query.sort(**{s: 'asc'})
    
    # Handle other parameters
    params = {}
    for key in ['page', 'per-page', 'sample', 'seed']:
        if key in query_params:
            params[key] = query_params[key][0]
    
    return query, params

def invert_abstract(inv_index):
    """Reconstruct abstract from OpenAlex' inverted-index.

    Handles dicts, JSON / repr strings, or missing values gracefully.
    """
    # Try to coerce a string into a Python object first
    if isinstance(inv_index, str):
        try:
            inv_index = json.loads(inv_index)          # double-quoted JSON
        except Exception:
            try:
                inv_index = ast.literal_eval(inv_index)  # single-quoted repr
            except Exception:
                inv_index = None

    if isinstance(inv_index, dict):
        l_inv = [(w, p) for w, pos in inv_index.items() for p in pos]
        return " ".join(w for w, _ in sorted(l_inv, key=lambda x: x[1]))
    else:
        return " "  
    
    
def get_pub(x):
    """Extract publication name from record."""
    try: 
        source = x['source']['display_name']
        if source not in ['parsed_publication','Deleted Journal']:
            return source
        else: 
            return ' '
    except:
            return ' '

def get_field(x):
    """Extract academic field from record."""
    try:
        field = x['primary_topic']['subfield']['display_name']
        if field is not None:
            return field
        else:
            return np.nan
    except:
        return np.nan

def process_records_to_df(records):
    """
    Convert OpenAlex records to a pandas DataFrame with processed fields.
    Can handle either raw OpenAlex records or an existing DataFrame.
    
    Args:
    records (list or pd.DataFrame): List of OpenAlex record dictionaries or existing DataFrame
    
    Returns:
    pandas.DataFrame: Processed DataFrame with abstracts, publications, and titles
    """
    # If records is already a DataFrame, use it directly.
    if isinstance(records, pd.DataFrame):
        records_df = records.copy()
    else:
        records_df = pd.DataFrame(records)

    # ADDED (custom CSV support): normalize and clean column names.
    records_df.columns = [str(col).strip() for col in records_df.columns]

    # ADDED (custom CSV support): accept the common typo "abtract" as "abstract".
    if 'abstract' not in records_df.columns and 'abtract' in records_df.columns:
        records_df = records_df.rename(columns={'abtract': 'abstract'})

    # If we have OpenAlex inverted abstracts, reconstruct them. Existing abstracts are only
    # filled where missing/empty.
    if 'abstract_inverted_index' in records_df.columns:
        reconstructed_abstracts = pd.Series(
            [invert_abstract(t) for t in records_df['abstract_inverted_index']],
            index=records_df.index
        )
        if 'abstract' not in records_df.columns:
            records_df['abstract'] = reconstructed_abstracts
        else:
            abstract_series = records_df['abstract']
            missing_mask = abstract_series.isna() | (abstract_series.astype(str).str.strip() == '')
            records_df.loc[missing_mask, 'abstract'] = reconstructed_abstracts.loc[missing_mask]

    # Keep previous OpenAlex publication parsing behavior when available.
    if 'primary_location' in records_df.columns:
        records_df['parsed_publication'] = [get_pub(x) for x in records_df['primary_location']]
        records_df['parsed_publication'] = records_df['parsed_publication'].fillna(' ')

    # ADDED (custom CSV support): guarantee core columns exist for downstream pipeline.
    if 'title' not in records_df.columns:
        records_df['title'] = ' '
    if 'abstract' not in records_df.columns:
        records_df['abstract'] = ' '

    records_df['abstract'] = records_df['abstract'].fillna(' ')
    records_df['title'] = records_df['title'].fillna(' ')

    # ADDED (custom CSV support): create stable ids when no id column is provided.
    if 'id' not in records_df.columns:
        records_df['id'] = [f"uploaded_{i}" for i in range(len(records_df))]
    else:
        id_series = records_df['id'].astype(str)
        missing_id_mask = records_df['id'].isna() | (id_series.str.strip() == '')
        records_df.loc[missing_id_mask, 'id'] = [
            f"uploaded_{i}" for i in records_df.index[missing_id_mask]
        ]

    records_df['id'] = records_df['id'].astype(str)
    records_df = records_df.drop_duplicates(subset=['id']).reset_index(drop=True)
    
    return records_df

def openalex_url_to_filename(url):
    """
    Convert an OpenAlex URL to a filename-safe string with timestamp.
    
    Args:
    url (str): The OpenAlex search URL
    
    Returns:
    str: A filename-safe string with timestamp (without extension)
    """
    from datetime import datetime
    import re
    
    # First parse the URL into query and params
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    
    # Create parts of the filename
    parts = []
    
    # Handle filters
    if 'filter' in query_params:
        filters = query_params['filter'][0].split(',')
        for f in filters:
            if ':' in f:
                key, value = f.split(':', 1)
                # Replace dots with underscores and clean the value
                key = key.replace('.', '_')
                # Clean the value to be filename-safe and add spaces around words
                clean_value = re.sub(r'[^\w\s-]', '', value)
                # Replace multiple spaces with single space and strip
                clean_value = ' '.join(clean_value.split())
                # Replace spaces with underscores for filename
                clean_value = clean_value.replace(' ', '_')
                
                if key == 'default_search':
                    parts.append(f"search_{clean_value}")
                else:
                    parts.append(f"{key}_{clean_value}")
    
    # Handle sort parameters
    if 'sort' in query_params:
        sort_params = query_params['sort'][0].split(',')
        for s in sort_params:
            if s.startswith('-'):
                parts.append(f"sort_{s[1:].replace('.', '_')}_desc")
            else:
                parts.append(f"sort_{s.replace('.', '_')}_asc")
    
    # Add timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Combine all parts
    filename = '__'.join(parts) if parts else 'openalex_query'
    filename = f"{filename}__{timestamp}"
    
    # Ensure filename is not too long (max 255 chars is common filesystem limit)
    if len(filename) > 255:
        filename = filename[:251]  # leave room for potential extension
    
    return filename

def get_records_from_dois(doi_list, block_size=50):
    """
    Download OpenAlex records for a list of DOIs in blocks.
    Args:
        doi_list (list): List of DOIs (strings)
        block_size (int): Number of DOIs to fetch per request (default 50)
    Returns:
        pd.DataFrame: DataFrame of OpenAlex records
    """
    from pyalex import Works
    from tqdm import tqdm
    all_records = []
    for i in tqdm(range(0, len(doi_list), block_size)):
        sublist = doi_list[i:i+block_size]
        doi_str = "|".join(sublist)
        try:
            record_list = Works().filter(doi=doi_str).get(per_page=block_size)
            all_records.extend(record_list)
        except Exception as e:
            print(f"Error fetching DOIs {sublist}: {e}")
    return pd.DataFrame(all_records)

def openalex_url_to_readable_name(url):
    """
    Convert an OpenAlex URL to a short, human-readable query description.
    
    Args:
    url (str): The OpenAlex search URL
    
    Returns:
    str: A short, human-readable description of the query
    
    Examples:
    - "Search: 'Kuramoto Model'"
    - "Search: 'quantum physics', 2020-2023"
    - "Cites: Popper (1959)"
    - "From: University of Pittsburgh, 1999-2020"
    - "By: Einstein, A., 1905-1955"
    """
    import re
    
    # Parse the URL
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    
    # Initialize description parts
    parts = []
    year_range = None
    
    # Handle filters
    if 'filter' in query_params:
        filters = query_params['filter'][0].split(',')
        
        for f in filters:
            if ':' not in f:
                continue
                
            key, value = f.split(':', 1)
            
            try:
                if key == 'default.search':
                    # Clean up search term (remove quotes if present)
                    search_term = value.strip('"\'')
                    parts.append(f"Search: '{search_term}'")
                    
                elif key == 'title_and_abstract.search':
                    # Handle title and abstract search specifically
                    from urllib.parse import unquote_plus
                    search_term = unquote_plus(value).strip('"\'')
                    parts.append(f"T&A: '{search_term}'")
                    
                elif key == 'publication_year':
                    # Handle year ranges or single years
                    if '-' in value:
                        start_year, end_year = value.split('-')
                        year_range = f"{start_year}-{end_year}"
                    else:
                        year_range = value
                        
                elif key == 'cites':
                    # Look up the cited work to get author and year
                    work_id = value
                    try:
                        cited_work = Works()[work_id]
                        if cited_work:
                            # Get first author's last name
                            author_name = "Unknown"
                            year = "Unknown"
                            
                            if cited_work.get('authorships') and len(cited_work['authorships']) > 0:
                                first_author = cited_work['authorships'][0]['author']
                                if first_author.get('display_name'):
                                    # Extract last name (assuming "First Last" format)
                                    name_parts = first_author['display_name'].split()
                                    author_name = name_parts[-1] if name_parts else first_author['display_name']
                            
                            if cited_work.get('publication_year'):
                                year = str(cited_work['publication_year'])
                                
                            parts.append(f"Cites: {author_name} ({year})")
                        else:
                            parts.append(f"Cites: Work {work_id}")
                    except Exception as e:
                        print(f"Could not fetch cited work {work_id}: {e}")
                        parts.append(f"Cites: Work {work_id}")
                        
                elif key == 'authorships.institutions.lineage':
                    # Look up institution name
                    inst_id = value
                    try:
                        institution = Institutions()[inst_id]
                        if institution and institution.get('display_name'):
                            parts.append(f"From: {institution['display_name']}")
                        else:
                            parts.append(f"From: Institution {inst_id}")
                    except Exception as e:
                        print(f"Could not fetch institution {inst_id}: {e}")
                        parts.append(f"From: Institution {inst_id}")
                        
                elif key == 'authorships.author.id':
                    # Look up author name
                    author_id = value
                    try:
                        author = Authors()[author_id]
                        if author and author.get('display_name'):
                            parts.append(f"By: {author['display_name']}")
                        else:
                            parts.append(f"By: Author {author_id}")
                    except Exception as e:
                        print(f"Could not fetch author {author_id}: {e}")
                        parts.append(f"By: Author {author_id}")
                        
                elif key == 'type':
                    # Handle work types
                    type_mapping = {
                        'article': 'Articles',
                        'book': 'Books',
                        'book-chapter': 'Book Chapters',
                        'dissertation': 'Dissertations',
                        'preprint': 'Preprints'
                    }
                    work_type = type_mapping.get(value, value.replace('-', ' ').title())
                    parts.append(f"Type: {work_type}")
                    
                elif key == 'host_venue.id':
                    # Look up venue name
                    venue_id = value
                    try:
                        # For venues, we can use Works to get source info, but let's try a direct approach
                        # This might need adjustment based on pyalex API structure
                        parts.append(f"In: Venue {venue_id}")  # Fallback
                    except Exception as e:
                        parts.append(f"In: Venue {venue_id}")
                        
                elif key.startswith('concepts.id'):
                    # Handle concept filters - these are topic/concept IDs
                    concept_id = value
                    parts.append(f"Topic: {concept_id}")  # Could be enhanced with concept lookup
                    
                else:
                    # Generic handling for other filters
                    from urllib.parse import unquote_plus
                    clean_key = key.replace('_', ' ').replace('.', ' ').title()
                    # Properly decode URL-encoded values
                    try:
                        clean_value = unquote_plus(value).replace('_', ' ')
                    except:
                        clean_value = value.replace('_', ' ')
                    parts.append(f"{clean_key}: {clean_value}")
                    
            except Exception as e:
                print(f"Error processing filter {f}: {e}")
                continue
    
    # Combine parts into final description
    if not parts:
        description = "OpenAlex Query"
    else:
        description = ", ".join(parts)
    
    # Add year range if present
    if year_range:
        if parts:
            description += f", {year_range}"
        else:
            description = f"Works from {year_range}"
    
    # Limit length to keep it readable
    if len(description) > 60:
        description = description[:57] + "..."
        
    return description
