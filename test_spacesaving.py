from crick import SpaceSaving

# Give each event a numeric ID
event_to_id = {
    "LOGIN_SUCCESS": 1.0,
    "FILE_ACCESS": 2.0,
    "LOGIN_FAILED": 3.0,
    "PROCESS_START": 4.0,
}

id_to_event = {v: k for k, v in event_to_id.items()}

stream = [
    "LOGIN_SUCCESS",
    "LOGIN_SUCCESS",
    "FILE_ACCESS",
    "LOGIN_SUCCESS",
    "LOGIN_FAILED",
    "FILE_ACCESS",
    "LOGIN_SUCCESS",
    "PROCESS_START",
    "FILE_ACCESS",
    "LOGIN_SUCCESS",
]

# Keep 3 counters
ss = SpaceSaving(3)

# Stream events into Space-Saving
for event in stream:
    ss.update(event_to_id[event])

# Get top 3
results = ss.topk(3)

print("Heavy hitters:")
print("Rank | Event          | Estimated Frequency | Error")
print("-" * 55)

for rank, (item, frequency, error) in enumerate(results, start=1):
    event = id_to_event[float(item)]
    print(f"{rank:4} | {event:14} | {frequency:19} | {error}")
