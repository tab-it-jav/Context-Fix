import sys, time
sys.path.insert(0,'.')
from engine import *
cf=ContextFix()
CASES=[
("i recieve teh book yesterday and it was definately good. She go to a apple store with there friends.\n\nHe dont like the the movie because it was to long. This is better then that, and the cat chased it's toy.",
 "I receive the book yesterday and it was definitely good. She goes to an apple store with their friends.\n\nHe doesn't like the movie because it was too long. This is better than that, and the cat chased its toy."),
("I ate to much food at lunch.","I ate too much food at lunch."),
("Their going to present their research.","They're going to present their research."),
("Your going to enjoy the show.","You're going to enjoy the show."),
("I am going too college.","I am going to college."),
("This book is better then that one.","This book is better than that one."),
("The wheather is nice today, I dont know weather it will rain.","The weather is nice today, I don't know whether it will rain."),
("Yesterday we goes to the park and they was happy. I has a dog. She dont like it.","Yesterday we go to the park and they were happy. I have a dog. She doesn't like it."),
("I would of gone but I couldnt find an umbrella. Alot of peopel were their.","I would have gone but I couldn't find an umbrella. A lot of people were there."),
("She walked to the libary and askd the librarian for a new storry.","She walked to the library and asked the librarian for a new story."),
("I want to go to the store to by some bread, but its to late.","I want to go to the store to buy some bread, but it's too late."),
("Teh quick brwon fox jumpd over teh lazy dgo.","The quick brown fox jumped over the lazy dog."),
("There was a big partty and alot of peopel came becuase it was her birthdya.","There was a big party and a lot of people came because it was her birthday."),
("She dont know wich way to go, so she askd me for adress.","She doesn't know which way to go, so she asked me for address."),
# these must stay UNCHANGED
("Let it go. Does she go there? Where does it work? I told you he likes it. Can he swim?",None),
("The principal said that their team lost its game, but it's okay because they're learning. Who's ready to go? Whose turn is it?",None),
("He is taller than me, and then we went to the park to play. It was too cold, but we had two dogs with us.",None),
("\"Whose boat is that?\" he whispered. They're still friends and it's better to help. \"Well, it's time,\" she said.",None),
("Please call me at john@gmail.com or visit www.example.com today. Dr. Smith lives in Hyderabad.",None),
]
def run(**kw):
    ok=0
    for src,exp in CASES:
        a=cf.analyze(src, **kw); out=apply_issues(src,a.issues,set())
        exp = src if exp is None else exp
        if out==exp: ok+=1
        else: print('  FAIL', repr(src[:60]),'\n   got',repr(out[:160]),'\n   exp',repr(exp[:160]))
    return ok
if __name__=='__main__':
    for lam in (2,5,10,20):
        print('lam',lam, run(lam=lam),'/',len(CASES))
