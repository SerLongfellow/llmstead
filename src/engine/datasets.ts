import { DatasetOption } from '../types';

/** Small seeded PRNG (mulberry32) so generated datasets are identical on every load */
function seededRandom(seed: number): () => number {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle<T>(items: T[], rand: () => number): T[] {
  const a = items.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/**
 * The math & logic dataset, generated rather than typed out. The original hand-written lines
 * come first (so the benchmark's "seen" cases stay in the training split); the rest are shuffled
 * with a fixed seed. Lines matching the benchmark's held-out cases are deliberately left out,
 * so those cases can only pass by generalizing: 2 + 1, 4 + 3 and 1 + 3 (commutativity),
 * the D → E rule chain, and cows.
 */
function buildMathLogicText(): string {
  const intro = [
    '1 + 1 = 2', '1 + 2 = 3', '2 + 2 = 4', '2 + 3 = 5', '3 + 3 = 6', '3 + 4 = 7', '4 + 4 = 8', '5 + 5 = 10',
    'If A then B. A is true. Therefore B.',
    'If B then C. B is true. Therefore C.',
    'If C then D. C is true. Therefore D.',
    'Cat is an animal. Animal has four legs. Cat has four legs.',
    'Dog is an animal. Animal has four legs. Dog has four legs.',
  ];
  const excluded = new Set([...intro, '2 + 1 = 3', '4 + 3 = 7', '1 + 3 = 4']);

  const sums: string[] = [];
  for (let a = 0; a <= 9; a++) {
    for (let b = 0; b <= 9; b++) {
      const line = `${a} + ${b} = ${a + b}`;
      if (!excluded.has(line)) sums.push(line);
    }
  }

  const comparisons: string[] = [];
  for (let a = 0; a <= 9; a++) {
    for (let b = 0; b <= 9; b++) {
      if (a > b && (a + b) % 3 === 0) comparisons.push(`${a} is greater than ${b}.`);
      if (a < b && (a + b) % 3 === 1) comparisons.push(`${a} is less than ${b}.`);
    }
  }

  const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  const rules: string[] = [];
  for (let i = 0; i < letters.length - 1; i++) {
    const [p, q] = [letters[i], letters[i + 1]];
    const line = `If ${p} then ${q}. ${p} is true. Therefore ${q}.`;
    if (p !== 'D' && !excluded.has(line)) rules.push(line); // D → E is held out
  }
  for (let i = 0; i < letters.length - 3; i += 2) {
    const [p, q] = [letters[i], letters[i + 3]];
    if (p !== 'D') rules.push(`If ${p} then ${q}. ${p} is true. Therefore ${q}.`);
  }

  const legged = ['Horse', 'Lion', 'Tiger', 'Sheep', 'Goat', 'Pig', 'Bear', 'Wolf', 'Fox', 'Rabbit', 'Mouse', 'Deer', 'Zebra', 'Camel'];
  const birds = ['Robin', 'Eagle', 'Owl', 'Duck', 'Crow', 'Parrot', 'Swan', 'Hawk'];
  const fish = ['Salmon', 'Trout', 'Shark', 'Tuna', 'Cod'];
  const facts = [
    ...legged.map(x => `${x} is an animal. Animal has four legs. ${x} has four legs.`),
    ...birds.map(x => `${x} is a bird. Bird has wings. ${x} has wings.`),
    ...fish.map(x => `${x} is a fish. Fish has fins. ${x} has fins.`),
  ];

  const rand = seededRandom(42);
  return [...intro, ...shuffle([...sums, ...comparisons, ...rules, ...facts], rand)].join('\n');
}

export const SAMPLE_DATASETS: DatasetOption[] = [
  {
    id: 'shakespeare',
    name: 'Tiny Shakespeare',
    category: 'literature',
    description: 'Coriolanus, Act 1 Scenes 1 and 3 (public domain) for learning archaic English vocabulary & rhythm.',
    text: `FIRST CITIZEN:
Before we proceed any further, hear me speak.

ALL:
Speak, speak.

FIRST CITIZEN:
You are all resolved rather to die than to famish?

ALL:
Resolved, resolved.

FIRST CITIZEN:
First, you know Caius Marcius is chief enemy to the people.

ALL:
We know't, we know't.

FIRST CITIZEN:
Let us kill him, and we'll have corn at our own price. Is't a verdict?

ALL:
No more talking on't; let it be done: away, away!

SECOND CITIZEN:
One word, good citizens.

FIRST CITIZEN:
We are accounted poor citizens, the patricians good. What authority surfeits on would relieve us: if they would yield us but the superfluity, while it were wholesome, we might think they relieved us humanely; but they think we are too dear: the leanness that afflicts us, the object of our misery, is as an inventory to particularise their abundance; our sufferance is a gain to them. Let us revenge this with our pikes, ere we become rakes: for the gods know I speak this in hunger for bread, not in thirst for revenge.

SECOND CITIZEN:
Would you proceed especially against Caius Marcius?

FIRST CITIZEN:
Against him first: he's a very dog to the commonalty.

SECOND CITIZEN:
Consider you what services he has done for his country?

FIRST CITIZEN:
Very well; and could be content to give him good report for't, but that he pays himself with being proud.

SECOND CITIZEN:
Nay, but speak not maliciously.

FIRST CITIZEN:
I say unto you, what he hath done famously, he did it to that end: though soft-conscienced men can be content to say it was for his country, he did it to please his mother, and to be partly proud; which he is, even till the altitude of his virtue.

SECOND CITIZEN:
What he cannot help in his nature, you account a vice in him. You must in no way say he is covetous.

FIRST CITIZEN:
If I must not, I need not be barren of accusations; he hath faults, with surplus, to tire in repetition. What shouts are these? The other side o' the city is risen: why stay we prating here? to the Capitol!

ALL:
Come, come.

FIRST CITIZEN:
Soft! who comes here?

MENENIUS:
What work's, my countrymen, in hand? where go you With bats and clubs? The matter? speak, I pray you.

FIRST CITIZEN:
Our business is not unknown to the senate; they have had inkling this fortnight what we intend to do, which now we'll show 'em in deeds. They say poor suitors have strong breaths: they shall know we have strong arms too.

MENENIUS:
Why, masters, my good friends, my honest neighbours, Will you undo yourselves?

FIRST CITIZEN:
We cannot, sir, we are undone already.

MENENIUS:
I tell you, friends, most charitable care Have the patricians for you. For your wants, Your suffering in this dearth, you may as well Strike at the heaven with your staves as lift them Against the Roman state, whose course will on The way it takes, cracking ten thousand curbs Of more strong link asunder than can ever Appear in your impediment. For the dearth, The gods, not the patricians, make it, and Your knees to them, not arms, must help. Alack, You slander the helms o' the state, who care for you Like fathers, when you curse them as enemies.

FIRST CITIZEN:
Care for us! True, indeed! They ne'er cared for us yet: suffer us to famish, and their store-houses crammed with grain; make edicts for usury, to support usurers; repeal any wholesome act established against the rich, and provide more piercing statutes daily, to chain up and restrain the poor. If the wars eat us not up, they will; and there's all the love they bear us.

MENENIUS:
Either you must Confess yourselves wondrous malicious, Or be accused of folly. I shall tell you A pretty tale: it may be you have heard it; But, since it serves my purpose, I will venture To stale 't a little more.

FIRST CITIZEN:
Well, I'll hear it, sir: yet you must not think to fob off our disgrace with a tale: but, an 't please you, deliver.

MENENIUS:
There was a time when all the body's members Rebelled against the belly, thus accused it: That only like a gulf it did remain I' the midst o' the body, idle and unactive, Still cupboarding the viand, never bearing Like labour with the rest, where the other instruments Did see and hear, devise, instruct, walk, feel, And, mutually participate, did minister Unto the appetite and affection common Of the whole body. The belly answered--

FIRST CITIZEN:
Well, sir, what answer made the belly?

MENENIUS:
Sir, I shall tell you. With a kind of smile, Which ne'er came from the lungs, but even thus-- For, look you, I may make the belly smile As well as speak--it tauntingly replied To the discontented members, the mutinous parts That envied his receipt; even so most fitly As you malign our senators for that They are not such as you.

FIRST CITIZEN:
Your belly's answer? What! The kingly-crowned head, the vigilant eye, The counsellor heart, the arm soldier, Our steed the leg, the tongue our trumpeter, With other muniments and petty helps In this our fabric, if that they--

MENENIUS:
What then? 'Fore me, this fellow speaks! What then? what then?

FIRST CITIZEN:
Should by the cormorant belly be restrain'd, Who is the sink o' the body,--

MENENIUS:
Well, what then?

FIRST CITIZEN:
The former agents, if they did complain, What could the belly answer?

MENENIUS:
I will tell you; If you'll bestow a small--of what you have little-- Patience awhile, you'll hear the belly's answer.

FIRST CITIZEN:
Ye're long about it.

MENENIUS:
Note me this, good friend; Your most grave belly was deliberate, Not rash like his accusers, and thus answer'd: 'True is it, my incorporate friends,' quoth he, 'That I receive the general food at first, Which you do live upon; and fit it is, Because I am the store-house and the shop Of the whole body: but, if you do remember, I send it through the rivers of your blood, Even to the court, the heart, to the seat o' the brain; And, through the cranks and offices of man, The strongest nerves and small inferior veins From me receive that natural competency Whereby they live: and though that all at once, You, my good friends,'--this says the belly, mark me,--

FIRST CITIZEN:
Ay, sir; well, well.

MENENIUS:
'Though all at once cannot See what I do deliver out to each, Yet I can make my audit up, that all From me do back receive the flour of all, And leave me but the bran.' What say you to't?

FIRST CITIZEN:
It was an answer: how apply you this?

MENENIUS:
The senators of Rome are this good belly, And you the mutinous members; for examine Their counsels and their cares, digest things rightly Touching the weal o' the common, you shall find No public benefit which you receive But it proceeds or comes from them to you And no way from yourselves. What do you think, You, the great toe of this assembly?

FIRST CITIZEN:
I the great toe! why the great toe?

MENENIUS:
For that, being one o' the lowest, basest, poorest, Of this most wise rebellion, thou go'st foremost: Thou rascal, that art worst in blood to run, Lead'st first to win some vantage. But make you ready your stiff bats and clubs: Rome and her rats are at the point of battle; The one side must have bale. Hail, noble Marcius!

MARCIUS:
Thanks. What's the matter, you dissentious rogues, That, rubbing the poor itch of your opinion, Make yourselves scabs?

FIRST CITIZEN:
We have ever your good word.

MARCIUS:
He that will give good words to thee will flatter Beneath abhorring. What would you have, you curs, That like nor peace nor war? the one affrights you, The other makes you proud. He that trusts to you, Where he should find you lions, finds you hares; Where foxes, geese: you are no surer, no, Than is the coal of fire upon the ice, Or hailstone in the sun. Your virtue is To make him worthy whose offence subdues him And curse that justice did it. Who deserves greatness Deserves your hate; and your affections are A sick man's appetite, who desires most that Which would increase his evil. He that depends Upon your favours swims with fins of lead And hews down oaks with rushes. Hang ye! Trust ye? With every minute you do change a mind, And call him noble that was now your hate, Him vile that was your garland. What's the matter, That in these several places of the city You cry against the noble senate, who, Under the gods, keep you in awe, which else Would feed on one another? What's their seeking?

MENENIUS:
For corn at their own rates; whereof, they say, The city is well stored.

MARCIUS:
Hang 'em! They say! They'll sit by the fire, and presume to know What's done i' the Capitol; who's like to rise, Who thrives and who declines; side factions and give out Conjectural marriages; making parties strong And feebling such as stand not in their liking Below their cobbled shoes. They say there's grain enough! Would the nobility lay aside their ruth, And let me use my sword, I'll make a quarry With thousands of these quarter'd slaves, as high As I could pick my lance.

MENENIUS:
Nay, these are almost thoroughly persuaded; For though abundantly they lack discretion, Yet are they passing cowardly. But, I beseech you, What says the other troop?

MARCIUS:
They are dissolved: hang 'em! They said they were an-hungry; sigh'd forth proverbs, That hunger broke stone walls, that dogs must eat, That meat was made for mouths, that the gods sent not Corn for the rich men only: with these shreds They vented their complainings; which being answer'd, And a petition granted them, a strange one-- To break the heart of generosity, And make bold power look pale--they threw their caps As they would hang them on the horns o' the moon, Shouting their emulation.

MENENIUS:
What is granted them?

MARCIUS:
Five tribunes to defend their vulgar wisdoms, Of their own choice: one's Junius Brutus, Sicinius Velutus, and I know not--'Sdeath! The rabble should have first unroof'd the city, Ere so prevail'd with me: it will in time Win upon power and throw forth greater themes For insurrection's arguing.

MENENIUS:
This is strange.

MARCIUS:
Go, get you home, you fragments!

MESSENGER:
Where's Caius Marcius?

MARCIUS:
Here: what's the matter?

MESSENGER:
The news is, sir, the Volsces are in arms.

MARCIUS:
I am glad on 't: then we shall ha' means to vent Our musty superfluity. See, our best elders.

FIRST SENATOR:
Marcius, 'tis true that you have lately told us; The Volsces are in arms.

MARCIUS:
They have a leader, Tullus Aufidius, that will put you to 't. I sin in envying his nobility, And were I any thing but what I am, I would wish me only he.

COMINIUS:
You have fought together.

MARCIUS:
Were half to half the world by the ears and he Upon my party, I'ld revolt to make Only my wars with him: he is a lion That I am proud to hunt.

FIRST SENATOR:
Then, worthy Marcius, Attend upon Cominius to these wars.

COMINIUS:
It is your former promise.

MARCIUS:
Sir, it is; And I am constant. Titus Lartius, thou Shalt see me once more strike at Tullus' face. What, art thou stiff? stand'st out?

TITUS:
No, Caius Marcius; I'll lean upon one crutch and fight with t'other, Ere stay behind this business.

MENENIUS:
O, true-bred!

FIRST SENATOR:
Your company to the Capitol; where, I know, Our greatest friends attend us.

TITUS:
Lead you on. Follow Cominius; we must follow you; Right worthy you priority.

COMINIUS:
Noble Marcius!

FIRST SENATOR:
Hence to your homes; be gone!

MARCIUS:
Nay, let them follow: The Volsces have much corn; take these rats thither To gnaw their garners. Worshipful mutiners, Your valour puts well forth: pray, follow.

SICINIUS:
Was ever man so proud as is this Marcius?

BRUTUS:
He has no equal.

SICINIUS:
When we were chosen tribunes for the people,--

BRUTUS:
Mark'd you his lip and eyes?

SICINIUS:
Nay, but his taunts.

BRUTUS:
Being moved, he will not spare to gird the gods.

SICINIUS:
Be-mock the modest moon.

BRUTUS:
The present wars devour him: he is grown Too proud to be so valiant.

SICINIUS:
Such a nature, Tickled with good success, disdains the shadow Which he treads on at noon: but I do wonder His insolence can brook to be commanded Under Cominius.

BRUTUS:
Fame, at the which he aims, In whom already he's well graced, can not Better be held nor more attain'd than by A place below the first: for what miscarries Shall be the general's fault, though he perform To the utmost of a man, and giddy censure Will then cry out of Marcius 'O if he Had borne the business!'

SICINIUS:
Besides, if things go well, Opinion that so sticks on Marcius shall Of his demerits rob Cominius.

BRUTUS:
Come: Half all Cominius' honours are to Marcius, Though Marcius earn'd them not, and all his faults To Marcius shall be honours, though indeed In aught he merit not.

SICINIUS:
Let's hence, and hear How the dispatch is made, and in what fashion, More than his singularity, he goes Upon this present action.

BRUTUS:
Let's along.

VOLUMNIA:
I pray you, daughter, sing; or express yourself in a more comfortable sort: if my son were my husband, I should freelier rejoice in that absence wherein he won honour than in the embracements of his bed where he would show most love. When yet he was but tender-bodied and the only son of my womb, when youth with comeliness plucked all gaze his way, when for a day of kings' entreaties a mother should not sell him an hour from her beholding, I, considering how honour would become such a person, that it was no better than picture-like to hang by the wall, if renown made it not stir, was pleased to let him seek danger where he was like to find fame. To a cruel war I sent him; from whence he returned, his brows bound with oak. I tell thee, daughter, I sprang not more in joy at first hearing he was a man-child than now in first seeing he had proved himself a man.

VIRGILIA:
But had he died in the business, madam; how then?

VOLUMNIA:
Then his good report should have been my son; I therein would have found issue. Hear me profess sincerely: had I a dozen sons, each in my love alike and none less dear than thine and my good Marcius, I had rather had eleven die nobly for their country than one voluptuously surfeit out of action.

GENTLEWOMAN:
Madam, the Lady Valeria is come to visit you.

VIRGILIA:
Beseech you, give me leave to retire myself.

VOLUMNIA:
Indeed, you shall not. Methinks I hear hither your husband's drum, See him pluck Aufidius down by the hair, As children from a bear, the Volsces shunning him: Methinks I see him stamp thus, and call thus: 'Come on, you cowards! you were got in fear, Though you were born in Rome:' his bloody brow With his mail'd hand then wiping, forth he goes, Like to a harvest-man that's task'd to mow Or all or lose his hire.

VIRGILIA:
His bloody brow! O Jupiter, no blood!

VOLUMNIA:
Away, you fool! it more becomes a man Than gilt his trophy: the breasts of Hecuba, When she did suckle Hector, look'd not lovelier Than Hector's forehead when it spit forth blood At Grecian sword, contemning. Tell Valeria, We are fit to bid her welcome.

VIRGILIA:
Heavens bless my lord from fell Aufidius!

VOLUMNIA:
He'll beat Aufidius' head below his knee And tread upon his neck.`
  },
  {
    id: 'math-logic',
    name: 'Synthetic Math & Logic',
    category: 'logic',
    description: 'Single-digit sums, if-then rule chains and category facts, generated so a few benchmark cases never appear.',
    text: buildMathLogicText(),
  },
  {
    id: 'code-python',
    name: 'Python Micro Snippets',
    category: 'code',
    description: 'Short Python functions, loops, conditionals and classes for learning code structure.',
    text: `def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

for i in range(5):
    print(i)

x = 10
if x > 5:
    print("Greater")
else:
    print("Smaller")

def divide(a, b):
    return a / b

def power(a, b):
    return a ** b

def maximum(a, b):
    if a > b:
        return a
    return b

def minimum(a, b):
    if a < b:
        return a
    return b

def average(a, b):
    return (a + b) / 2

def mod(a, b):
    return a % b

def concat(a, b):
    return a + b

def square(x):
    return x * x

def cube(x):
    return x * x * x

def double(x):
    return x * 2

def half(x):
    return x / 2

def negate(x):
    return -x

def is_even(n):
    return n % 2 == 0

def greet(name):
    print("Hello, " + name)

for i in range(10):
    print(i * 2)

for i in range(3):
    print("Hello")

for k in range(4):
    print(k)

for i in range(2, 8):
    print(i)

for i in range(6):
    if i % 2 == 0:
        print(i)

total = 0
for i in range(5):
    total = total + i
print(total)

n = 7
if n > 3:
    print("Big")
else:
    print("Small")

temperature = 30
if temperature > 25:
    print("Hot")
else:
    print("Cold")

age = 18
if age > 17:
    print("Adult")
else:
    print("Minor")

z = 0
if z > 0:
    print("Positive")
elif z < 0:
    print("Negative")
else:
    print("Zero")

a = 3
b = 4
if a > b:
    print("a is bigger")
else:
    print("b is bigger")

count = 0
while count < 5:
    print(count)
    count = count + 1

names = ["Ann", "Bob", "Cat"]
for name in names:
    print(name)

numbers = [1, 2, 3, 4, 5]
for n in numbers:
    if n % 2 == 0:
        print("even")
    else:
        print("odd")

scores = {"Ann": 90, "Bob": 85}
for key in scores:
    print(key, scores[key])

word = "hello"
for letter in word:
    print(letter)

def sum_list(items):
    total = 0
    for item in items:
        total = total + item
    return total

def count_words(text):
    return len(text.split())

def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

def clamp(x, low, high):
    if x < low:
        return low
    if x > high:
        return high
    return x

result = add(2, 3)
print(result)

value = multiply(4, 5)
print(value)

class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value = self.value + 1

    def get(self):
        return self.value

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def add(self, other):
        return Point(self.x + other.x, self.y + other.y)

for i in range(100):
    if i > 3:
        break
    print(i)

def first(items):
    return items[0]

def last(items):
    return items[-1]

m = 12
if m > 10:
    print("Double digits")
else:
    print("Single digit")`
  },
  {
    id: 'qa-dialogue',
    name: 'Simple Q&A Conversations',
    category: 'synthetic',
    description: 'User/Assistant turns: greetings plus short definitions of ML terms.',
    text: `User: Hello!
Assistant: Hi there! How can I help you today?

User: What is an LLM?
Assistant: An LLM is a Large Language Model trained to predict tokens.

User: What is attention?
Assistant: Attention allows models to dynamically focus on relevant tokens.

User: Hey!
Assistant: Hi! What would you like to know?

User: Good morning!
Assistant: Hi! I hope your day is going well.

User: What is a tokenizer?
Assistant: A tokenizer splits text into tokens the model can read.

User: What is a transformer?
Assistant: A transformer is a neural network built from attention and MLP layers.

User: What is an embedding?
Assistant: An embedding is a list of numbers that represents a token.

User: What is a neural network?
Assistant: A neural network is a stack of layers that learns patterns from data.

User: What is training?
Assistant: Training adjusts the model's weights so its predictions get better.

User: What is a loss?
Assistant: A loss is a number that measures how wrong the model's predictions are.

User: What is a gradient?
Assistant: A gradient tells each weight which direction lowers the loss.

User: What is backpropagation?
Assistant: Backpropagation computes the gradient of the loss for every weight.

User: What is a weight?
Assistant: A weight is a number inside the model that is learned during training.

User: What is a vocabulary?
Assistant: A vocabulary is the fixed set of tokens a model knows.

User: What is softmax?
Assistant: Softmax turns a list of scores into probabilities that add up to one.

User: What is a context window?
Assistant: A context window is the number of tokens the model can see at once.

User: What is an MLP?
Assistant: An MLP is a small feed-forward network inside each transformer block.

User: What is a layer?
Assistant: A layer is one step of computation inside a neural network.

User: Thanks!
Assistant: You're welcome! Happy to help.

User: What is overfitting?
Assistant: Overfitting is when a model memorizes its training data instead of learning general patterns.

User: What is a benchmark?
Assistant: A benchmark is a set of tests used to measure a model.

User: What is a prompt?
Assistant: A prompt is the text you give a model to continue.

User: What is temperature?
Assistant: Temperature controls how random the model's choices are.

User: What is an optimizer?
Assistant: An optimizer uses gradients to update the model's weights.

User: What is a head?
Assistant: A head is one of several parallel attention units in a layer.

User: Hello again!
Assistant: Hi! What else can I help with?

User: What is a parameter?
Assistant: A parameter is another name for a learned weight.

User: What is fine-tuning?
Assistant: Fine-tuning continues training a model on a smaller, specific dataset.

User: What is a dataset?
Assistant: A dataset is the collection of text a model learns from.

User: Can you explain attention?
Assistant: Attention lets each token look back at earlier tokens and gather the useful ones.

User: How does a model generate text?
Assistant: It predicts one token, adds it to the input, and repeats.

User: Why do models use tokens?
Assistant: Tokens turn text into numbers the model can compute with.

User: Thank you!
Assistant: You're welcome!

User: What is an epoch?
Assistant: An epoch is one full pass over the training data.

User: What is a logit?
Assistant: A logit is a raw score for a token before softmax.

User: Goodbye!
Assistant: Goodbye! Have a great day.`
  }
];

/** Fraction of each dataset (by characters, snapped to a line break) held out from training. */
export const VALIDATION_FRACTION = 0.15;

export interface DatasetSplit {
  trainText: string;
  valText: string;
}

/**
 * Split a dataset into a training portion and a held-out validation tail.
 * The model never trains on valText, so loss on it measures generalization
 * rather than memorization.
 */
export function splitDataset(text: string, valFraction: number = VALIDATION_FRACTION): DatasetSplit {
  const target = Math.floor(text.length * (1 - valFraction));
  // Prefer a paragraph break (keeps Q&A turns / code blocks intact), then a line break
  const minCut = Math.floor(text.length * (1 - 3 * valFraction));
  let cut = text.lastIndexOf('\n\n', target);
  if (cut < minCut) cut = text.lastIndexOf('\n', target);
  if (cut <= 0) cut = target; // no line break before the mark — cut mid-line
  return {
    trainText: text.slice(0, cut),
    valText: text.slice(cut).replace(/^\n+/, ''),
  };
}
