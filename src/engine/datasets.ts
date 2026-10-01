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
 * Python dataset: the hand-written snippets, then templated variations (functions, loops,
 * conditionals) with a fixed shuffle. The templates never produce the benchmark's held-out
 * cases: no `subtract` function, no loop over `j`, and no conditionals on a variable `y`.
 */
function buildCodeText(handwritten: string): string {
  const rand = seededRandom(7);
  const pick = <T,>(items: T[]) => items[Math.floor(rand() * items.length)];
  const snippets: string[] = [];

  const binary: [string, string][] = [
    ['add_numbers', '+'], ['plus', '+'], ['sum_two', '+'], ['combine', '+'], ['times', '*'], ['product', '*'],
    ['scale', '*'], ['minus', '-'], ['difference', '-'], ['quotient', '/'], ['ratio', '/'], ['remainder', '%'],
    ['floor_div', '//'], ['raise_to', '**'],
  ];
  const argPairs: [string, string][] = [['a', 'b'], ['x', 'z'], ['m', 'n'], ['left', 'right'], ['first', 'second']];
  for (const [name, op] of binary) {
    for (const [p, q] of shuffle(argPairs, rand).slice(0, 2)) {
      snippets.push(`def ${name}(${p}, ${q}):\n    return ${p} ${op} ${q}`);
    }
  }

  const unary: [string, string][] = [
    ['add_one', '+ 1'], ['add_two', '+ 2'], ['add_ten', '+ 10'], ['minus_one', '- 1'], ['triple', '* 3'],
    ['quadruple', '* 4'], ['tenfold', '* 10'], ['third', '/ 3'], ['halve', '/ 2'], ['squared', '** 2'], ['last_digit', '% 10'],
  ];
  for (const [name, rest] of unary) {
    const v = pick(['x', 'n', 'value', 'num']);
    snippets.push(`def ${name}(${v}):\n    return ${v} ${rest}`);
  }

  const loopVars = ['i', 'k', 'n', 'x', 'idx', 'step'];
  for (let t = 0; t < 30; t++) {
    const v = pick(loopVars);
    const expr = pick([v, `${v} * 2`, `${v} + 1`, `${v} * ${v}`, `${v} - 1`, '"hi"', '"tick"']);
    snippets.push(`for ${v} in range(${1 + Math.floor(rand() * 12)}):\n    print(${expr})`);
  }

  const condVars = ['x', 'n', 'a', 'b', 'count', 'score', 'total', 'speed', 'level', 'size', 'price'];
  const labels: [string, string][] = [['Big', 'Small'], ['High', 'Low'], ['Yes', 'No'], ['Pass', 'Fail'], ['Many', 'Few'], ['Over', 'Under']];
  for (let t = 0; t < 28; t++) {
    const v = pick(condVars);
    const [hi, lo] = pick(labels);
    const threshold = Math.floor(rand() * 50);
    const value = Math.floor(rand() * 100);
    snippets.push(`${v} = ${value}\nif ${v} > ${threshold}:\n    print("${hi}")\nelse:\n    print("${lo}")`);
  }

  for (let t = 0; t < 10; t++) {
    const v = pick(['count', 'i', 'k', 'tries', 'steps']);
    snippets.push(`${v} = 0\nwhile ${v} < ${2 + Math.floor(rand() * 8)}:\n    print(${v})\n    ${v} = ${v} + 1`);
  }

  const lists: [string, string, string][] = [
    ['fruits', 'fruit', '["apple", "pear", "plum"]'], ['colors', 'color', '["red", "green", "blue"]'],
    ['pets', 'pet', '["cat", "dog", "fish"]'], ['nums', 'num', '[3, 1, 4, 1, 5]'], ['words', 'word', '["one", "two", "three"]'],
    ['primes', 'p', '[2, 3, 5, 7, 11]'],
  ];
  for (const [list, item, values] of lists) {
    snippets.push(`${list} = ${values}\nfor ${item} in ${list}:\n    print(${item})`);
  }

  return handwritten + '\n\n' + shuffle(snippets, rand).join('\n\n');
}

/**
 * Q&A dataset: the hand-written turns, then generated ones: every "What is X?" definition
 * asked a second way, extra definitions, and more greetings and thanks. Nothing generated
 * asks "What is a token?" or greets with "Hi!" (both held-out benchmark cases).
 */
function buildQaText(handwritten: string): string {
  const rand = seededRandom(11);
  const turns: string[] = [];
  const turn = (q: string, a: string) => `User: ${q}\nAssistant: ${a}`;

  const extraDefinitions: [string, string][] = [
    ['an activation function', 'An activation function adds a nonlinear bend between layers, like GELU or ReLU.'],
    ['GELU', 'GELU is a smooth activation function used in transformer MLPs.'],
    ['a residual connection', "A residual connection adds a layer's input back to its output."],
    ['layer normalization', 'Layer normalization rescales each vector to zero mean and unit variance.'],
    ['a query', 'A query is the vector a token uses to look for relevant tokens.'],
    ['a key', 'A key is the vector a token offers so that others can find it.'],
    ['a value', 'A value is the information a token passes along when it is attended to.'],
    ['a learning rate', 'A learning rate sets how big each weight update is.'],
    ['a batch', 'A batch is a group of examples processed together in one step.'],
    ['validation loss', 'Validation loss is the loss on text the model never trained on.'],
    ['a hyperparameter', 'A hyperparameter is a setting chosen before training, like the learning rate.'],
    ['AdamW', 'AdamW is an optimizer that adapts the step size for every weight.'],
    ['greedy decoding', 'Greedy decoding always picks the most likely next token.'],
    ['sampling', 'Sampling picks the next token at random, weighted by the probabilities.'],
    ['perplexity', 'Perplexity is the exponential of the loss, roughly how many choices the model is torn between.'],
    ['a merge', 'A merge combines two frequent neighbors into one new vocabulary entry.'],
    ['BPE', 'BPE builds a vocabulary by repeatedly merging the most frequent pair.'],
    ['a causal mask', 'A causal mask stops each position from looking at later positions.'],
    ['positional encoding', 'Positional encoding tells the model where each position sits in the sequence.'],
    ['inference', 'Inference is running a trained model to make predictions.'],
    ['RLHF', 'RLHF fine-tunes a model using human feedback on its answers.'],
    ['a GPU', 'A GPU is a chip that does many calculations in parallel.'],
    ['a probability', 'A probability is a number between zero and one.'],
    ['an attention head', 'An attention head is one of several parallel attention units in a layer.'],
  ];
  const definitions: [string, string][] = [];
  for (const m of handwritten.matchAll(/User: What is (.+)\?\nAssistant: (.+)/g)) definitions.push([m[1], m[2]]);
  for (const [term, answer] of extraDefinitions) {
    turns.push(turn(`What is ${term}?`, answer));
    definitions.push([term, answer]);
  }

  // Ask every definition a second way
  const phrasings = [(t: string) => `Can you explain ${t}?`, (t: string) => `Tell me about ${t}.`, (t: string) => `Define ${t}.`];
  for (const [term, answer] of definitions) {
    const ask = phrasings[Math.floor(rand() * phrasings.length)];
    turns.push(turn(ask(term), answer));
  }

  const greetings = ['Hello there!', 'Hey there!', 'Good afternoon!', 'Good evening!', 'Howdy!', 'Greetings!', 'Yo!'];
  const replies = ['Hi! What would you like to know?', 'Hi there! Ask me anything about language models.', 'Hi! How can I help?'];
  greetings.forEach((g, i) => turns.push(turn(g, replies[i % replies.length])));
  const thanks = ['Thanks a lot!', 'Thank you so much!', 'That helps, thanks!', 'Great, thank you!'];
  thanks.forEach(t => turns.push(turn(t, "You're welcome!")));

  return handwritten + '\n\n' + shuffle(turns, rand).join('\n\n');
}

/**
 * The math & logic dataset, generated rather than typed out. The original hand-written lines
 * come first (so the benchmark's "seen" cases stay in the training split); the rest are shuffled
 * with a fixed seed. Anything matching the benchmark's held-out cases is deliberately left out,
 * in every format, so those cases can only pass by generalizing: the sums 2 + 1, 4 + 3 and
 * 1 + 3 (commutativity), the D → E rule chain, and cows.
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
  const introSet = new Set(intro);
  const heldOutSum = (a: number, b: number) =>
    (a === 2 && b === 1) || (a === 4 && b === 3) || (a === 1 && b === 3);

  // Arithmetic
  const sums: string[] = [];
  for (let a = 0; a <= 9; a++) {
    for (let b = 0; b <= 9; b++) {
      if (heldOutSum(a, b)) continue;
      const line = `${a} + ${b} = ${a + b}`;
      if (!introSet.has(line)) sums.push(line);
      sums.push(`What is ${a} plus ${b}? ${a + b}.`);
    }
  }
  for (let a = 10; a <= 39; a++) {
    for (let b = 1; b <= 9; b++) {
      sums.push(`${a} + ${b} = ${a + b}`);
      if (a < 20) sums.push(`${b} + ${a} = ${a + b}`); // both orders, so order-independence is visible
    }
  }
  const differences: string[] = [];
  for (let a = 0; a <= 18; a++) {
    for (let b = Math.max(0, a - 9); b <= Math.min(a, 9); b++) {
      differences.push(`${a} - ${b} = ${a - b}`);
      if (a <= 9) differences.push(`What is ${a} minus ${b}? ${a - b}.`);
    }
  }
  const comparisons: string[] = [];
  for (let a = 0; a <= 9; a++) {
    for (let b = 0; b <= 9; b++) {
      if (a > b) comparisons.push(`${a} is greater than ${b}.`);
      if (a < b) comparisons.push(`${a} is less than ${b}.`);
    }
  }
  const parity: string[] = [];
  for (let n = 0; n <= 30; n++) parity.push(`${n} is ${n % 2 === 0 ? 'even' : 'odd'}.`);

  // Logic: one-step and two-step rule chains, skipping anything with D → E
  const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  const rules: string[] = [];
  const usesHeldOut = (...pairs: [string, string][]) => pairs.some(([p, q]) => p === 'D' && q === 'E');
  for (let i = 0; i < letters.length - 1; i++) {
    const [p, q] = [letters[i], letters[i + 1]];
    const line = `If ${p} then ${q}. ${p} is true. Therefore ${q}.`;
    if (!usesHeldOut([p, q]) && !introSet.has(line)) rules.push(line);
  }
  for (let i = 0; i < letters.length - 3; i++) {
    const [p, q] = [letters[i], letters[i + 3]];
    if (!usesHeldOut([p, q])) rules.push(`If ${p} then ${q}. ${p} is true. Therefore ${q}.`);
  }
  for (let i = 0; i < letters.length - 2; i++) {
    const [p, q, r] = [letters[i], letters[i + 1], letters[i + 2]];
    if (!usesHeldOut([p, q], [q, r])) rules.push(`If ${p} then ${q}. If ${q} then ${r}. ${p} is true. Therefore ${r}.`);
  }

  // Category facts (no cows: "Cow" is a held-out benchmark case)
  const mammals = ['Horse', 'Lion', 'Tiger', 'Sheep', 'Goat', 'Pig', 'Bear', 'Wolf', 'Fox', 'Rabbit', 'Mouse', 'Deer', 'Zebra',
    'Camel', 'Elephant', 'Giraffe', 'Hippo', 'Rhino', 'Moose', 'Donkey', 'Llama', 'Panda', 'Leopard', 'Cheetah', 'Hamster',
    'Squirrel', 'Otter', 'Badger', 'Raccoon', 'Bison', 'Yak'];
  const birds = ['Robin', 'Eagle', 'Owl', 'Duck', 'Crow', 'Parrot', 'Swan', 'Hawk', 'Sparrow', 'Pigeon', 'Goose', 'Falcon', 'Heron', 'Finch'];
  const fish = ['Salmon', 'Trout', 'Shark', 'Tuna', 'Cod', 'Carp', 'Bass', 'Pike', 'Herring', 'Perch'];
  const reptiles = ['Snake', 'Lizard', 'Crocodile', 'Turtle', 'Iguana', 'Gecko'];
  const insects = ['Ant', 'Bee', 'Beetle', 'Wasp', 'Moth', 'Fly', 'Cricket', 'Grasshopper'];
  const facts = [
    ...mammals.map(x => `${x} is an animal. Animal has four legs. ${x} has four legs.`),
    ...birds.map(x => `${x} is a bird. Bird has wings. ${x} has wings.`),
    ...fish.map(x => `${x} is a fish. Fish has fins. ${x} has fins.`),
    ...reptiles.map(x => `${x} is a reptile. Reptile has scales. ${x} has scales.`),
    ...insects.map(x => `${x} is an insect. Insect has six legs. ${x} has six legs.`),
  ];

  // Final safety net: drop any line that merely *contains* a held-out case, e.g. "32 + 1 = 33"
  // contains "2 + 1 = 3" (the benchmark's seen/held-out check matches substrings too).
  const HELD_OUT = ['2 + 1 = 3', '4 + 3 = 7', '1 + 3 = 4', 'If D then E', 'Cow'];
  const generated = [...sums, ...differences, ...comparisons, ...parity, ...rules, ...facts]
    .filter(line => !HELD_OUT.some(h => line.includes(h)));

  const rand = seededRandom(42);
  return [...intro, ...shuffle(generated, rand)].join('\n');
}

export const SAMPLE_DATASETS: DatasetOption[] = [
  {
    id: 'math-logic',
    name: 'Synthetic Math & Logic',
    samplePrompt: '3 + 4 = ',
    category: 'logic',
    description: 'Single-digit sums, if-then rule chains and category facts, generated so a few benchmark cases never appear.',
    text: buildMathLogicText(),
  },
  {
    id: 'shakespeare',
    name: 'Tiny Shakespeare',
    samplePrompt: 'FIRST CITIZEN:\n',
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
    id: 'code-python',
    name: 'Python Micro Snippets',
    samplePrompt: 'def add(a, b):\n    ',
    category: 'code',
    description: 'Short Python functions, loops, conditionals and classes for learning code structure.',
    text: buildCodeText(`def add(a, b):
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
    print("Single digit")`),
  },
  {
    id: 'qa-dialogue',
    name: 'Simple Q&A Conversations',
    samplePrompt: 'User: What is attention?\nAssistant: ',
    category: 'synthetic',
    description: 'User/Assistant turns: greetings plus short definitions of ML terms.',
    text: buildQaText(`User: Hello!
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
Assistant: Goodbye! Have a great day.`),
  }
];

/** A dataset's example prompt; custom datasets fall back to the start of their first line */
export const samplePromptFor = (ds: DatasetOption) => ds.samplePrompt ?? ds.text.split('\n')[0].slice(0, 30);

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
