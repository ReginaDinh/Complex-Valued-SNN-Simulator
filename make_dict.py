import json

with open('translations_todo.json') as f:
    todo = json.load(f)

print(len(todo))
