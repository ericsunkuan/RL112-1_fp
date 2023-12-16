import re

# Input string
input_string = "[3(0), 2, 1(30), 4, 2]"

# Define regular expressions for values with and without parentheses
value_pattern = re.compile(r'(\d+)(?:\((\d+)\))?')

# Find all matches in the input string
matches = re.findall(value_pattern, input_string)

# Initialize variables
values_without_parenthesis = []
values_with_parenthesis = []

# Extract values from matches
for match in matches:
    value_without_parenthesis = int(match[0])
    value_with_parenthesis = int(match[1]) if match[1] is not '' else 0

    values_without_parenthesis.append(value_without_parenthesis)
    values_with_parenthesis.append(value_with_parenthesis)

# Print the results
print("Values without parenthesis:", values_without_parenthesis)
print("Values with parenthesis:", values_with_parenthesis)

# Assign values to individual variables (optional)
var1, var2, var3, var4, var5 = values_without_parenthesis[:5]
var6, var7, var8, var9, var10 = values_with_parenthesis[:5]

# Print individual variables (optional)
print("Individual variables without parenthesis:", var1, var2, var3, var4, var5)
print("Individual variables with parenthesis:", var6, var7, var8, var9, var10)
