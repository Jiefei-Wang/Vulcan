import sys
import io
import ast
from typing import Dict, List
from contextlib import redirect_stdout

# Create a persistent global namespace for all code executions
PERSISTENT_GLOBALS = {}

def trace(x):
    print(x)


def execute_and_embed(filename: str, output_filename: str = None):
    """
    Execute Python file with #> markers and embed output, preserving whitespace.
    
    Args:
        filename: Input Python file
        output_filename: Output file (if None, overwrites input file)
    """
    global PERSISTENT_GLOBALS
    
    # Initialize with built-ins and current globals
    PERSISTENT_GLOBALS.update(globals())
    
    if output_filename is None:
        output_filename = filename
        
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at '{filename}'")
        return

    blocks = split_into_blocks(content)
    
    result_blocks = []
    encounter_error = False
    for block in blocks:
        if not encounter_error:
            result_lines, encounter_error = execute_block(block)
            result_blocks.append(result_lines)
        else:
            # If an error was encountered, just append the original lines
            result_blocks.append(block['lines'])
    
    # Update the main module's globals with variables from execution
    import __main__
    __main__.__dict__.update(PERSISTENT_GLOBALS)
    
    final_content = '\n'.join(['\n'.join(lines) for lines in result_blocks])
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(final_content)
        
    


def split_into_blocks(content: str) -> List[Dict]:
    """
    Splits content into blocks. A block is a chunk of code/comments/whitespace
    followed by a marker.
    This implementation preserves empty lines between blocks.
    """
    lines = content.split('\n')
    blocks = []
    current_lines = []
    
    for line in lines:
        is_output_line = line.strip().startswith('#>')
        if is_output_line:
            continue
        
        is_output_object = line.strip().startswith('trace(')
        
        if is_output_object:
            if current_lines:
                blocks.append({'lines': current_lines, 'has_output_marker': False})
                current_lines = []
            blocks.append({'lines': [line], 'has_output_marker': True})
        else:
            current_lines.append(line)
        
    if current_lines:
        blocks.append({'lines': current_lines, 'has_output_marker': False})
        
    return blocks


def execute_block(block: Dict) -> Dict:
    """
    Executes a code block, cleans old output, captures new output,
    and formats it cleanly without a leading empty marker.
    """
    global PERSISTENT_GLOBALS
    
    lines = block['lines']
    has_output_marker = block.get('has_output_marker', False)
    
    code = '\n'.join(lines)
    
    output = None
    encounter_error = False
    try:
        if not has_output_marker:
            exec(code, PERSISTENT_GLOBALS)
        else:
            f = io.StringIO()
            with redirect_stdout(f):
                exec(code, PERSISTENT_GLOBALS)
            output = f.getvalue()
            # remove the last \n if it exists
            if output.endswith('\n'):
                output = output[:-1]
    except Exception as e:
        encounter_error = True
        print(f"Error executing block: {e}")
        
        
    if output is not None:
        output_lines = output.split('\n')
        output_lines = [f"#> {line}" for line in output_lines]
    
    result_lines = lines + output_lines if output is not None else lines
    
    return result_lines, encounter_error

