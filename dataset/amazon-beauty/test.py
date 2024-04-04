
with open('test.txt','r') as f:
    lines=f.readlines()
    print(len(lines))
f1=open('test1.txt','w')
for i in range(len(lines)):
    line=lines[i]
    f1.write(str(line).replace(',',' ')+'')
f1.close
