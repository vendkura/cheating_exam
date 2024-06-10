import pandas as pd
import matplotlib.pyplot as plt
import os

log_dir = './logs'  # replace with your log directory path
log_file = 'detection_2024-05-27_23-34-23.csv'  # replace with your log file

df = pd.read_csv(os.path.join(log_dir, log_file), header=None)

# Rename the columns for clarity
df.columns = ['Timestamp', 'Unknown', 'Log Level', 'Message']

# Parse the timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Extract the confidence level from the message
df['Confidence'] = df['Message'].str.extract('confidence ([\d.]+)', expand=False).astype(float)

# Plot the data
plt.figure()  # Create a new figure for this plot
plt.plot(df['Timestamp'], df['Confidence'], label='Confidence')

# Add labels and legend
plt.xlabel('Timestamp')
plt.ylabel('Confidence')
plt.legend()

plt.show()
# Print the 'Message' data for the top 5 confidence values
top_confidence = df.nlargest(5, 'Confidence')
print(top_confidence['Message'])
