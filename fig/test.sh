#!/bin/bash



cat ./lexiconp.txt | awk '{print $1}' | sort | uniq  | awk '
  BEGIN {
    print "<eps> 0";
  }
  {
    if ($1 == "<s>") {
      print "<s> is in the vocabulary!" | "cat 1>&2"
      printf("*****************%s %d\n", $1, NR);
      exit 1;
    }
    if ($1 == "</s>") {
      print "</s> is in the vocabulary!" | "cat 1>&2"
      printf("*****************%s %d\n", $1, NR);
      exit 1;
    }
    printf("%s %d\n", $1, NR);
  }
  END {
    printf("#0 %d\n", NR+1);
    printf("<s> %d\n", NR+2);
    printf("</s> %d\n", NR+3);
  }' >words.txt || exit 1;
  
  
  cat ./lexiconp.txt | awk '{print $1}' | sort | uniq >words.txt || exit 1;
