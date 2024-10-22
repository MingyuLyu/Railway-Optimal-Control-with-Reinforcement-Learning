import matplotlib.pyplot as plt

# Test data
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

# Create a basic plot
plt.plot(x, y, label='Test plot')

# Add labels and a title
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Simple Matplotlib Plot')
plt.legend()

# Display the plot
plt.show()
