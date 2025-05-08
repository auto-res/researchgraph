"""
Verify Modules

This script checks that all the modules for the web enhanced paper search
can be imported correctly.
"""

import sys
import os
import importlib.util

def verify_modules():
    """Verify that all modules can be imported correctly"""
    print("Verifying modules...")
    
    # Get current directory and nodes directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    nodes_dir = os.path.join(current_dir, "nodes")
    
    print(f"Current directory: {current_dir}")
    print(f"Nodes directory: {nodes_dir}")
    
    # Add both directories to Python path
    sys.path.insert(0, current_dir)
    sys.path.insert(0, nodes_dir)
    
    # Define files to import
    module_files = [
        os.path.join(nodes_dir, "openai_web_search_node.py"),
        os.path.join(nodes_dir, "extract_papers_from_web_search.py"),
        os.path.join(nodes_dir, "web_based_paper_search.py"),
        os.path.join(nodes_dir, "web_enhanced_paper_search.py"),
        os.path.join(current_dir, "web_enhanced_subgraph.py")
    ]
    
    # Import each module using importlib
    success_count = 0
    for module_path in module_files:
        module_name = os.path.basename(module_path).replace(".py", "")
        try:
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print(f"✅ Successfully imported {module_name}")
                success_count += 1
            else:
                print(f"❌ Failed to load spec for {module_name}")
        except Exception as e:
            print(f"❌ Failed to import {module_name}: {str(e)}")
    
    print(f"\nSuccessfully imported {success_count} of {len(module_files)} modules")
    
    # Check that required packages can be imported
    try:
        from openai import OpenAI
        print("✅ Successfully imported OpenAI client")
    except ImportError as e:
        print(f"❌ Failed to import OpenAI client: {str(e)}")
    
    try:
        from pydantic import BaseModel
        print("✅ Successfully imported Pydantic")
    except ImportError as e:
        print(f"❌ Failed to import Pydantic: {str(e)}")
    
    try:
        import langgraph
        print("✅ Successfully imported LangGraph")
    except ImportError as e:
        print(f"❌ Failed to import LangGraph: {str(e)}")

    return success_count == len(module_files)


if __name__ == "__main__":
    success = verify_modules()
    if success:
        print("\n✅ All modules verified successfully")
    else:
        print("\n❌ Some modules failed verification")
