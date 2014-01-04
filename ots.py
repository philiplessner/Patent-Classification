from subprocess import call

infile = 'test.txt'
out = 'out.txt'
with open(out, 'w') as outfile:
    call(['ots', infile], stdout=outfile)
