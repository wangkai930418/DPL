with open('text_inv.sh') as f:
    lines = f.read()

lines = lines.replace('\\','').split()
lines='\"' + '\",\"'.join(lines) + '\"'
print(lines)