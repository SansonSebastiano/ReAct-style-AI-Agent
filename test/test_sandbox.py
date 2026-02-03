from src.tools.sandbox import CodeSandbox

def test_simple_execution():
    """Test basic code execution"""
    sandbox = CodeSandbox(timeout=10, output_dir="test_output")
    
    code = """
print("Hello from sandbox!")
result = 2 + 2
print(f"2 + 2 = {result}")
"""
    
    result = sandbox.execute_code(code)
    print("Test 1 - Simple execution:")
    print(f"  Success: {result['success']}")
    print(f"  Stdout: {result['stdout']}")
    print(f"  Stderr: {result['stderr']}")
    print()

def test_plotly_generation():
    """Test Plotly HTML generation"""
    sandbox = CodeSandbox(timeout=10, output_dir="test_output")
    
    code = """
import plotly.express as px
import pandas as pd

# Create sample data
df = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [2, 4, 6, 8, 10]
})

# Create plot
fig = px.line(df, x='x', y='y', title='Sample Line Plot')

# Save to output.html (sandbox expects this name)
fig.write_html('output.html')

print("Plot generated successfully")
"""
    
    result = sandbox.execute_code(code)
    print("Test 2 - Plotly generation:")
    print(f"  Success: {result['success']}")
    print(f"  Stdout: {result['stdout']}")
    print(f"  Plot path: {result['plot_path']}")
    print()

def test_timeout():
    """Test timeout handling"""
    sandbox = CodeSandbox(timeout=2, output_dir="test_output")
    
    code = """
import time
print("Starting long operation...")
time.sleep(10)  # Will timeout
print("This won't be printed")
"""
    
    result = sandbox.execute_code(code)
    print("Test 3 - Timeout:")
    print(f"  Success: {result['success']}")
    print(f"  Stderr: {result['stderr']}")
    print()

def test_error_handling():
    """Test error handling"""
    sandbox = CodeSandbox(timeout=10, output_dir="test_output")
    
    code = """
print("About to raise error...")
raise ValueError("Something went wrong!")
"""
    
    result = sandbox.execute_code(code)
    print("Test 4 - Error handling:")
    print(f"  Success: {result['success']}")
    print(f"  Stdout: {result['stdout']}")
    print(f"  Stderr: {result['stderr']}")
    print()

def test_data_analysis():
    """Test data analysis code"""
    sandbox = CodeSandbox(timeout=10, output_dir="test_output")
    
    code = """
import pandas as pd
import plotly.express as px

# Create dataset
df = pd.DataFrame({
    'category': ['A', 'B', 'C', 'D'],
    'values': [23, 45, 56, 78]
})

# Analyze
print("Data Summary:")
print(df.describe())

# Visualize
fig = px.bar(df, x='category', y='values', title='Category Values')
fig.write_html('output.html')

print("\\nAnalysis complete!")
"""
    
    result = sandbox.execute_code(code)
    print("Test 5 - Data analysis:")
    print(f"  Success: {result['success']}")
    print(f"  Stdout: {result['stdout']}")
    print(f"  Plot path: {result['plot_path']}")
    print()

if __name__ == "__main__":
    print("Running CodeSandbox tests...\n")
    print("=" * 60)
    
    test_simple_execution()
    test_plotly_generation()
    test_timeout()
    test_error_handling()
    test_data_analysis()
    
    print("=" * 60)
    print("Tests complete! Check 'test_output/' for generated plots.")