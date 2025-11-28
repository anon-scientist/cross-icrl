"""
Cancels all currently running and scheduled jobs by parsing the squeue output for job ids and printing them out for scancel
Invocation: squeue | python3 extract.py | xargs scancel
"""

i = 0 ;
while True:
  i += 1 ;
  s = "" ;
  try:
    s = input() ;
  except Exception:
    break ;
  s = [x.strip() for x in s.split(" ") if x.strip() != ""] ;
  if i > 1:
    print(s[0])
