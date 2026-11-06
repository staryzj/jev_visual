"""Original short and rationale prompts and label-independent parsers."""
import re


def parse_answer(text,k,candidates=None):
 text=text.strip()
 m=re.fullmatch(r'\s*\(?([A-Z])\)?[.)]?\s*',text)
 if not m:m=re.match(r'^(?:The\s+)?(?:correct\s+)?(?:answer|option)\s*(?:is|:)\s*\(?([A-Z])\)?(?:[.):\s]|$)',text,re.I)
 if not m and candidates:
  # Accept "C. woman" only when the suffix exactly names option C.
  # This uses candidates, never the gold label; conflicting answers stay invalid.
  v=re.fullmatch(r'\s*\(?([A-Z])\)?[.):]\s*(.+?)\s*',text)
  norm=lambda s:re.sub(r'\s+',' ',s).strip().strip('.').casefold()
  if v:
   i=ord(v[1])-65
   if 0<=i<k and norm(v[2])==norm(candidates[i]):return i
 if not m:return None
 v=ord(m[1].upper())-65
 return v if 0<=v<k else None
def question(row):
 # The same question text and full candidate list as the JEV confirmation.
 assert 1<len(row['candidates'])<=26
 return row['question']+'\n'+ '\n'.join(f'{chr(65+i)}. {s}' for i,s in enumerate(row['candidates']))+'\nAnswer with only a single option letter.'


def rationale_question(row):
    choices = '\n'.join(chr(65 + index) + '. ' + value for index, value in enumerate(row['candidates']))
    return row['question'] + '\n' + choices + '\nReason step by step using the available evidence. Then finish with a new line containing only "Final answer: X", replacing X with the single correct option letter.'


def parse_rationale(text, row):
    matches = list(re.finditer(r'(?:^|\n)\s*Final answer:\s*\(?([A-Z])\)?[.)]?\s*(?=$|\n)', text, re.I))
    if not matches:
        return None, None
    match = matches[-1]
    index = ord(match.group(1).upper()) - 65
    if not 0 <= index < len(row['candidates']):
        return None, None
    return index, match.start()


