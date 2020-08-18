import numpy as np
import pandas as pd
import os
from gensim.summarization.summarizer import summarize
from gensim.summarization.textcleaner import clean_text_by_sentences

import numpy as np
from sklearn.preprocessing import normalize


def pagerank(x, df=0.85, max_iter=30, bias=None):
    assert 0 < df < 1
    A = normalize(x, axis=0, norm='l1')
    R = np.ones(A.shape[0]).reshape(-1, 1)

    if bias is None:
        bias = (1 - df) * np.ones(A.shape[0]).reshape(-1, 1)
    else:
        bias = bias.reshape(-1, 1)
        bias = A.shape[0] * bias / bias.sum()
        assert bias.shape[0] == A.shape[0]
        bias = (1 - df) * bias

    for _ in range(max_iter):
        R = df * (A * R) + bias
    return R


from collections import Counter
import math
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances


def sent_graph(sents, tokenize=None, min_count=2, min_sim=0.3,
               similarity=None, vocab_to_idx=None, verbose=False):
    if vocab_to_idx is None:
        idx_to_vocab, vocab_to_idx = scan_vocabulary(sents, tokenize, min_count)
    else:
        idx_to_vocab = [vocab for vocab, _ in sorted(vocab_to_idx.items(), key=lambda x: x[1])]
    x = vectorize_sents(sents, tokenize, vocab_to_idx)
    if similarity == 'cosine':
        x = numpy_cosine_similarity_matrix(x, min_sim, verbose, batch_size=1000)
    else:
        x = numpy_textrank_similarity_matrix(x, min_sim, verbose, batch_size=1000)
    return x


def vectorize_sents(sents, tokenize, vocab_to_idx):
    rows, cols, data = [], [], []
    for i, sent in enumerate(sents):
        counter = Counter(tokenize(sent))
        for token, count in counter.items():
            j = vocab_to_idx.get(token, -1)
            if j == -1:
                continue
            rows.append(i)
            cols.append(j)
            data.append(count)
    n_rows = len(sents)
    n_cols = len(vocab_to_idx)
    return csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))


def numpy_cosine_similarity_matrix(x, min_sim=0.3, verbose=True, batch_size=1000):
    n_rows = x.shape[0]
    mat = []
    for bidx in range(math.ceil(n_rows / batch_size)):
        b = int(bidx * batch_size)
        e = min(n_rows, int((bidx + 1) * batch_size))
        psim = 1 - pairwise_distances(x[b:e], x, metric='cosine')
        rows, cols = np.where(psim >= min_sim)
        data = psim[rows, cols]
        mat.append(csr_matrix((data, (rows, cols)), shape=(e - b, n_rows)))
        if verbose:
            print('\rcalculating cosine sentence similarity {} / {}'.format(b, n_rows), end='')
    mat = sp.sparse.vstack(mat)
    if verbose:
        print('\rcalculating cosine sentence similarity was done with {} sents'.format(n_rows))
    return mat


def numpy_textrank_similarity_matrix(x, min_sim=0.3, verbose=True, min_length=1, batch_size=1000):
    n_rows, n_cols = x.shape

    rows, cols = x.nonzero()
    data = np.ones(rows.shape[0])
    z = csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

    size = np.asarray(x.sum(axis=1)).reshape(-1)
    size[np.where(size <= min_length)] = 10000
    size = np.log(size)

    mat = []
    for bidx in range(math.ceil(n_rows / batch_size)):

        # slicing
        b = int(bidx * batch_size)
        e = min(n_rows, int((bidx + 1) * batch_size))

        # dot product
        inner = z[b:e, :] * z.transpose()

        # sentence len[i,j] = size[i] + size[j]
        norm = size[b:e].reshape(-1, 1) + size.reshape(1, -1)
        norm = norm ** (-1)
        norm[np.where(norm == np.inf)] = 0

        # normalize
        sim = inner.multiply(norm).tocsr()
        rows, cols = (sim >= min_sim).nonzero()
        data = np.asarray(sim[rows, cols]).reshape(-1)

        # append
        mat.append(csr_matrix((data, (rows, cols)), shape=(e - b, n_rows)))

        if verbose:
            print('\rcalculating textrank sentence similarity {} / {}'.format(b, n_rows), end='')

    mat = sp.sparse.vstack(mat)
    if verbose:
        print('\rcalculating textrank sentence similarity was done with {} sents'.format(n_rows))

    return mat


class KeysentenceSummarizer:

    def __init__(self, sents=None, tokenize=None, min_count=2,
                 min_sim=0.3, similarity=None, vocab_to_idx=None,
                 df=0.85, max_iter=30, verbose=False):

        self.tokenize = tokenize
        self.min_count = min_count
        self.min_sim = min_sim
        self.similarity = similarity
        self.vocab_to_idx = vocab_to_idx
        self.df = df
        self.max_iter = max_iter
        self.verbose = verbose

        if sents is not None:
            self.train_textrank(sents)

    def train_textrank(self, sents, bias=None):
        g = sent_graph(sents, self.tokenize, self.min_count,
                       self.min_sim, self.similarity, self.vocab_to_idx, self.verbose)
        self.R = pagerank(g, self.df, self.max_iter, bias).reshape(-1)
        if self.verbose:
            print('trained TextRank. n sentences = {}'.format(self.R.shape[0]))

    def summarize(self, sents, topk=30, bias=None):
        n_sents = len(sents)
        if isinstance(bias, np.ndarray):
            if bias.shape != (n_sents,):
                raise ValueError('The shape of bias must be (n_sents,) but {}'.format(bias.shape))
        elif bias is not None:
            raise ValueError('The type of bias must be None or numpy.ndarray but the type is {}'.format(type(bias)))

        self.train_textrank(sents, bias)
        idxs = self.R.argsort()[-topk:]
        keysents = [sents[idx] for idx in reversed(idxs)]
        return keysents


def scan_vocabulary(sents, tokenize=None, min_count=2):
    counter = Counter(w for sent in sents for w in tokenize(sent))
    counter = {w: c for w, c in counter.items() if c >= min_count}
    idx_to_vocab = [w for w, _ in sorted(counter.items(), key=lambda x: -x[1])]
    vocab_to_idx = {vocab: idx for idx, vocab in enumerate(idx_to_vocab)}

    return idx_to_vocab, vocab_to_idx


from nltk.tokenize import sent_tokenize

from gensim.summarization.summarizer import summarize

import nltk


def komoran_tokenizer(sent):
    words = nltk.tokenize.word_tokenize(sent)

    words = nltk.tag.pos_tag(words)

    return words


import networkx
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


class RawSentence:
    def __init__(self, textIter):
        if type(textIter) == str:
            self.textIter = textIter.split('\n')
        else:
            self.textIter = textIter
        self.rgxSplitter = re.compile('([.!?:](?:["\']|(?![0-9])))')

    def __iter__(self):
        for line in self.textIter:
            ch = self.rgxSplitter.split(line)
            for s in map(lambda a, b: a + b, ch[::2], ch[1::2]):
                if not s: continue
                yield s


class TextRank:
    def __init__(self, **kargs):
        self.graph = None
        self.window = kargs.get('window', 5)
        self.coef = kargs.get('coef', 1.0)
        self.threshold = kargs.get('threshold', 0.005)
        self.dictCount = {}
        self.dictBiCount = {}
        self.dictNear = {}
        self.nTotal = 0

    def loadSents(self, sentenceIter, tokenizer=None):
        import math
        def similarity(a, b):
            n = len(a.intersection(b))
            return n / float(len(a) + len(b) - n) / (math.log(len(a) + 1) * math.log(len(b) + 1))

        if not tokenizer: rgxSplitter = re.compile('[\\s.,:;-?!()"\']+')
        sentSet = []

        for sent in filter(None, sentenceIter):
            if type(sent) == str:
                if tokenizer:
                    s = set(filter(None, tokenizer(sent)))
                else:
                    s = set(filter(None, rgxSplitter.split(sent)))

            else:
                s = set(sent)
            if len(s) < 2: continue
            self.dictCount[len(self.dictCount)] = sent
            sentSet.append(s)

        for i in range(len(self.dictCount)):
            for j in range(i + 1, len(self.dictCount)):
                s = similarity(sentSet[i], sentSet[j])
                if s < self.threshold: continue
                self.dictBiCount[i, j] = s

    def build(self):
        self.graph = networkx.Graph()
        self.graph.add_nodes_from(self.dictCount.keys())
        for (a, b), n in self.dictBiCount.items():
            self.graph.add_edge(a, b, weight=n * self.coef + (1 - self.coef))

    def rank(self):
        return networkx.pagerank(self.graph, weight='weight')

    def summarize(self, lines=3):
        r = self.rank()
        ks = sorted(r, key=r.get, reverse=True)[:lines]
        return ' '.join(map(lambda k: self.dictCount[k], sorted(ks)))


stop_words = set(stopwords.words("english"))


def preprocessing(text):
    tokens = [word for sent in nltk.sent_tokenize(text)
              for word in nltk.word_tokenize(sent)]

    tokens = [token for token in tokens if token not in stop_words]

    tokens = [word for word in tokens if len(word) >= 3]

    tokens = [word.lower() for word in tokens]

    lmtzr = WordNetLemmatizer()
    tokens = [lmtzr.lemmatize(word) for word in tokens]
    tokens = [lmtzr.lemmatize(word, 'v') for word in tokens]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


import re


def samesent(a, b):
    text1 = re.sub('[^a-zA-Z0-9]', ' ', a).strip()
    text2 = re.sub('[^a-zA-Z0-9]', ' ', b).strip()
    text1 = text1.replace(" ", "")
    text2 = text2.replace(" ", "")
    if (text1 == text2):
        return True
    else:
        return False


def mixlist(lista, listb):
    lenb = len(listb)
    lena = len(lista)
    n = 0
    for i in range(lenb):
        for j in range(lena):
            if (samesent(lista[j], listb[i])):
                n = 1
            else:
                continue
        if (n == 0):
            lista.append(listb[i])
        n = 0
    return lista


def mcountlist(mainlist, sublist, countlist):
    mainlen = len(mainlist)
    sublen = len(sublist)
    n = 0
    for i in range(mainlen):
        for j in range(sublen):
            if (samesent(mainlist[i], sublist[j])):
                n = 1
            else:
                continue
        if (n == 1):
            countlist[i] = countlist[i] + 1
        n = 0
    return countlist


input1 = "After the bullet shells get counted, the blood dries and the votive candles burn out, people peer down from   windows and see crime scenes gone cold: a band of yellow police tape blowing in the breeze. The South Bronx, just across the Harlem River from Manhattan and once shorthand for urban dysfunction, still suffers violence at levels long ago slashed in many other parts of New York City. And yet the city?셲 efforts to fight it remain splintered, underfunded and burdened by scandal. In the 40th Precinct, at the southern tip of the Bronx, as in other poor, minority neighborhoods across the country, people long hounded for   infractions are crying out for more protection against grievous injury or death. By September, four of every five shootings in the precinct this year were unsolved. Out of the city?셲 77 precincts, the 40th has the highest murder rate but the fewest detectives per violent crime, reflecting disparities in staffing that hit hardest in some neighborhoods outside Manhattan, according to a New York Times analysis of Police Department data. Investigators in the precinct are saddled with twice the number of cases the department recommends, even as their bosses are called to Police Headquarters to answer for the sharpest crime rise in the city this year. And across the Bronx, investigative resources are squeezed. It has the highest   rate of the city?셲 five boroughs but the thinnest detective staffing. Nine of the 14   precinct detective squads for violent crime in the city are there. The borough?셲 robbery squad is smaller than Manhattan?셲, even though the Bronx has had 1, 300 more cases this year. And its homicide squad has one detective for every four murders, compared with one detective for roughly every two murders in Upper Manhattan and more than one detective per murder in Lower Manhattan. In   lobbies and   family apartments, outside methadone clinics and art studios, people take note of the inequity. They hear police commanders explain that they lack the resources to place a floodlight on a dangerous block or to post officers at a   corner. They watch witnesses cower behind   doors, more fearful of a gunman?셲 crew than confident in the Police Department?셲 ability to protect them. So though people see a lot, they rarely testify. And in the South Bronx, as in so many predominantly black and Hispanic neighborhoods like it in the United States, the contract between the police and the community is in tatters. Some people have stories of crime reports that were ignored, or 911 calls that went unanswered for hours. Others tell of a 911 call for help ending in the caller?셲 arrest, or of a minor charge leading to 12 hours in a fetid holding cell. This is the paradox of policing in the 40th Precinct. Its neighborhoods have historically been prime targets for aggressive tactics, like    that are designed to ward off disorder. But precinct detectives there have less time than anywhere else in the city to answer for the blood spilled in violent crimes. Gola White, who was beside her daughter when she was shot and killed in a playground this summer, four years after her son was gunned down in the same housing project, ticked off the public safety resources that she said were scant in Bronx neighborhoods like hers: security cameras, lights, locks, investigating police officers. ?쏦ere, we have nothing,??she said. When it comes to ?? families,??she said, the authorities ?쐂on?셳 really care as much. That?셲 how I feel. ??The Times has been documenting the murders logged this year in the 40th Precinct, one of a handful of neighborhoods where deadly violence remains a problem in an era of   crime in New York City. The homicides  ??  14 in the precinct this year, up from nine in 2015  ??  strain detectives, and when they go unsolved, as half of them have this year, some look to take the law into their own hands. From hundreds of conversations with grieving relatives and friends, witnesses and police officers, the social forces that flare into murder in a place like the 40th Precinct become clearer: merciless gang codes, mental illness, drugs and long memories of feuds that simmered out of officers??view. The reasons some murders will never be solved also emerge: paralyzing fear of retribution, victims carrying secrets to their graves and relentless casework that forces detectives to move on in hopes that a break will come later. Frustrations build on all sides. Detectives??phones rarely ring with tips, and officers grow embittered with witnesses who will not cooperate. In the meantime, a victim?셲 friends conduct their own investigations, and talk of grabbing a stash gun from a wheel well or a mother?셲 apartment when they find their suspect. In the chasm between the police and the community, gangs and gun violence flourish. Parents try to protect their families from drug crews??threats, and officers work to overcome the residue of years of mistrust and understaffing in communities where they still go racing from one 911 call to the next. The streets around St. Mary?셲 Park were the scene of two fatal shootings logged in the 40th Precinct this year. Both are unsolved. James Fernandez heard talk of the murders through the door of his   apartment on East 146th Street in a     the Betances Houses. He lived at the end of a long hallway strewn with hypodermic needles, empty dope bags and discarded Hennessy bottles. A   young men who spoke of being in a subset of the Bloods gang had made it their drug market, slinging marijuana and cocaine to regulars, flashing firearms and blowing smoke into the Fernandez apartment. When Mr. Fernandez, 40, asked the young men to move, they answered by busting up his car. This kind of crime, an anachronism in much of New York, still rattles the 40th Precinct, even though murders there have fallen to 14 this year from 83 in 1991. It has more major felony crimes per resident than any other residential district in the city. It is also one of the poorest communities in the country, and many young men find their way into underground markets. Mr. Fernandez was not one to shrink from the threats. When he was growing up on the Lower East Side, he rode his bicycle around to the customers of the drug dealers he worked for and collected payments in a backpack. After leaving that life, he got a tech maintenance job and, three years ago, moved into the Betances Houses with his wife and daughter, now 11. He had two choices to get help with the drug crew: call the police for help and risk being labeled a snitch, or call his old Lower East Side bosses for muscle and risk violence. He chose the police. Again and again, he walked into a local substation, Police Service Area 7, and asked for protection. His daughter was using an inhaler to relieve coughs from the marijuana smoke. Mr. Fernandez and his wife got terrible headaches. ?쏷here?셲 a lot of killers here, and we are going to kill you,??a sergeant?셲 police report quoted a    telling Mr. Fernandez in August 2015. A second report filed the same day said a    warned him, ?쏧?셫 going to shoot through your window. ??Mr. Fernandez told the police both the teenagers??names, which appear in the reports, and then went home. He said one of their friends had seen him walk into the substation, and they tried to intimidate him out of filing another report. Three days later, the same    propped his bike on their door, ?쐔hen said if I was to open the door and say something, they would body slam me,??Mr. Fernandez?셲 wife, Maria Fernandez, wrote on slips of paper she used to document the hallway ruckus and the inadequate police response. The boys made comments about how easy a target she was and about how they would have to ?쐓lap??her if she opened the door while they made a drug sale, and they threatened to beat the Fernandez family because ?쐔hey are the ones snitching,??her notes say. But another   complaint at the substation, 10 days after the first, brought no relief. A week later, feeling desperate, Ms. Fernandez tried calling: first to the substation, at 8:50 p. m. when one of the boys blew weed smoke at her door and made a   threat to attack her, and then to 911 at 10:36 p. m. The police never came, she wrote in her notes. She tried the 40th Precinct station house next, but officers at the desk left her standing in the public waiting area for a   she said, making her fear being seen again. Officers put her in worse danger some months later, she said, when they came to her door and announced in front of the teenagers that they were there on a complaint about drug activity. Mr. Fernandez started doing the work that he said the police had failed to do. He wired a camera into his peephole to record the drugs and guns. The footage hark back to the New York of the 1980s, still very much present to some of the precinct?셲 residents. Around 6:30 each morning, Sgt. Michael J. LoPuzzo walks through the tall wooden doors of the 40th Precinct station house. The cases that land on his metal desk  ??  dead bodies with no known cause, strip club brawls, shooting victims hobbling into the hospital themselves  ??  bring resistance at every turn, reminding him of an earlier era in the city?셲   campaign. ?쏧 haven?셳 got one single phone call that?셲 putting me in the right direction here,??said Sergeant LoPuzzo, the head of the precinct?셲 detective squad, one day this summer as he worked on an answer to an email inquiry from a murder victim?셲 aunt about why the killer had not been caught. ?쏛nd people just don?셳 understand that. ??Often it is detectives who most feel the effects of people turning on the police. Witnesses shout them away from their doors just so neighbors know they refuse to talk. Of the 184 people who were shot and wounded in the Bronx through early September, more than a third  ??  66 victims  ??  refused to cooperate. Over the same period in the 40th Precinct, squad detectives closed three of 17 nonfatal shootings, and 72 of 343 robbery cases. Part of the resistance stems from   preventive policing tactics, like    that were a hallmark of the     style under former Mayor Michael R. Bloomberg and his police commissioner, Raymond W. Kelly. Near the height of the    strategy, in 2012, the 40th Precinct had the   stops in the city, the   stops in which officers used force and the most frisks. Of 18, 276 stops that year, 15, 521 were of people who had done nothing criminal. The precinct was also one of the   areas that the department flooded with its newest officers. At roll calls, they were pressured to generate numbers: write tickets and make arrests. They had no choice but to give a summons to a young man playing in a park after dark, even if the officers had done the same growing up in the same neighborhood. ?쏧 need to bring something in today to justify my existence,??Officer Argenis Rosado, who joined the precinct in 2010, said in an interview at the station house. ?쏶o now you?셱e in a small area, and day after day you?셱e hammering the same community. Of course that community?셲 eventually going to turn on you. ??The pressure warped the way officers and residents saw each other. Rookies had to ignore why someone might be drinking outside or sitting on a stoop. ?쏶ome of the cops that came out at that time probably viewed the community differently, too,??said Hector Espada, a   veteran of the precinct. ?쏯ot because they wanted to, but because they had to. Because some way or somehow, you can?셳 give someone a $115 summons and feel like you guys could still have a civil conversation after that. ??Morale wilted in the aged station house on Alexander Avenue, in Mott Haven. Officers felt pressure to downgrade crime complaints to make them appear less serious. Several said in interviews that they had overlooked crime reports from immigrants because they were seen as unlikely to complain, and watched supervisors badger victims into repeating their stories in hopes that they would drop their complaints. The practice of downgrading complaints resulted in the disciplining of 19 officers in the precinct last year, one in a string of scandals that has left officers there feeling overscrutinized for problems that also existed elsewhere. Four commanders in the precinct were sent packing in five years, one of them after officers were found to be ?쐔icket fixing,??or forgiving parking tickets for friends, and another after he was recorded giving guidance on whom to stop and frisk: black boys and men, ages 14 to 21. Some officers fled to other commands. Others became reluctant to take assignments in proactive policing units, like   that put them in   situations on the street. ?쏻henever I walked through the doors of the precinct, to me, it seemed like a black cloud,??said Russell Lewis, a    of the 40th. ?쏧t was like a heaviness. When you walked in, all you wanted to do was do your 8 hours 35 minutes and go home, because you didn?셳 want to get caught up in anything. ??The precinct covers only about two square miles, but the more than a dozen housing projects there mean that it overflows with people. Methadone clinics draw addicts from around the city.   lofts on the southern edge of the precinct presage a wave of gentrification. Even as the Police Department has hired 1, 300 more officers for neighborhood policing and counterterrorism, officers in the 40th Precinct said they could still rush to 25 911 calls during a shift  ??  a number unchanged from what the new police commissioner, James P. O?셄eill, said he was handling in a similar South Bronx precinct 15 years ago. Several dozen calls at a time can be waiting for a response. Residents know that if you want the police for a domestic problem, it helps to hint that there is a weapon. Last year, the precinct drew the   number of civilian complaints for officer misconduct in the city, and the most lawsuits stemming from police actions. The precinct is trying to improve morale under a new commanding officer, Deputy Inspector Brian Hennessy. A cadre of what the department calls neighborhood coordination officers has been on patrol since last January, part of a citywide effort under Mr. O?셄eill and Mayor Bill de Blasio to bring back the beat cop, unencumbered by chasing every last 911 call, who can listen to people?셲 concerns and help with investigations. The precinct has made among the most gun arrests in the city, and officers said they now had more discretion to resolve encounters without a summons or an arrest. At one corner near a school, on Courtlandt Avenue and East 151st Street, that has long spawned complaints about gunfire and fights, Inspector Hennessy and some of his officers painted over graffiti and swept up drug paraphernalia this summer. People said it was the first answer to their complaints in years. But the inspector acknowledged that the residue of   policing lingers. ?쏷hat perception really sticks,??he said. The workload in the 40th Precinct is startling and reveals a gap in how detective squads are equipped to answer violent crime in Manhattan compared with the Bronx, Brooklyn and Queens. Three of the precinct?셲 16 detectives are carrying more than 400 cases each this year, and many others have loads in the high 300s, even though the department advises 150 in violent precincts. When they are assigned a homicide, they typically have four days to investigate before dealing with other cases. Quieter precincts can give detectives a month with little distraction to investigate a murder. Detectives in the 40th Precinct have each handled an average of 79 violent felonies this year through    ??  murders, rapes, felony assaults and robberies. By contrast, a detective in the precinct on the southern end of Staten Island carries nine such cases a detective in the precinct patrolling Union Square and Gramercy Park handles 16 and a detective in the precinct for most of Washington Heights handles 32, the citywide median. Last year, the 40th was the    for violent crime, with 65 cases per detective. In the Bronx as a whole, a precinct detective has carried an average of 58 violent felonies this year, compared with 27 in Manhattan, 37 in Brooklyn, 38 in Queens and 25 on Staten Island. Rape cases and robbery patterns are later sent to more specialized units, but precinct detectives do extensive initial work to interview victims, write reports and process evidence. Precincts in much of Manhattan, which are whiter and wealthier than the South Bronx, often have more property felonies, like stolen laptops or credit cards, and the police say those can be complex. But even accounting for those crimes, the 40th Precinct has some of the heaviest caseloads of overall crime per detective in the city. Michael Palladino, the head of the Detectives??Endowment Association and a former Bronx officer, said staffing disparities affected the department?셲 efforts to build trust in communities like the South Bronx. Witnesses make a calculation, he said: ?쏧f I cooperate with the detectives, there?셲 so much work, there?셲 so few of them there, they won?셳 even get the chance to protect me, or they?셪l be there too late when the retaliation comes. ??Sergeant LoPuzzo, who turned down a more prestigious post to stay in the 40th Precinct, said that his squad worked tirelessly to handle cases with the people he had, and that while every squad wanted more detectives, staffing needs for counterterrorism units and task forces had created new deployment challenges across the department. ?쏻e fight with the army we have, not the army we wish we had,??he said. Details of how the Police Department assigns its 36, 000 officers are closely held and constantly in flux, and the public has minimal information on how personnel are allocated. Presented with The Times?셲 analysis of confidential staffing data, the department?셲 chief of detectives, Robert K. Boyce, vowed to send more detectives to the 40th Precinct and said the department would reassess its deployment more broadly in troubled precincts. He said a recent decision to bring gang, narcotics and vice detectives under his command made it easier to shift personnel. Chief Boyce said the burdens on detectives went beyond felony crimes to include   and   cases. And he noted the support that precinct squads got from centralized units focusing on robberies, gangs or grand larcenies, for example. Major crime keeps pounding the 40th Precinct, at rates that in 2015 were only a tenth of a percent lower than in 2001, even as citywide crime dropped by more than a third over the same period. But the precinct?셲 detective squad shrank by about eight investigators during those years, according to staffing data obtained from the City Council through a Freedom of Information Law request. The squad covering Union Square and Gramercy Park, where crime dropped by a third over that period, grew by about 11 investigators. (The 40th Precinct was given an additional detective and four   investigators this summer, when it was already missing three detectives for illness or other reasons.) Retired detectives are skeptical that community relations alone can drive down crime in the city?셲 last ?쒋?the busiest precincts. Rather, they say, the Police Department should be dedicating more resources to providing the same sort of robust investigative response that seems standard in Manhattan. ?쏛ny crime in Manhattan has to be solved,??said Howard Landesberg, who was a 40th Precinct detective in the late 1980s. ?쏷he outer boroughs are, like, forgotten. ??Retired detectives said that understaffing made it harder to solve crimes in the Bronx, Brooklyn and Queens, where the higher prevalence of gang and drug killings already saddled investigators with cases in which people were not inclined to cooperate. Through   detectives had closed 67 percent of homicides in Manhattan and 76 percent of those in Staten Island this year, compared with 54 percent of those in the Bronx, 42 percent of those in Queens and 31 percent of those in Brooklyn. Of last year?셲 homicides, detectives cleared 71 percent in Manhattan, 63 percent in the Bronx, 62 percent in Queens, 57 percent in Staten Island and 31 percent in Brooklyn. ?쏧t?셲 the culture of the Police Department that they worry about Manhattan,??said Joseph L. Giacalone, a former sergeant in the Bronx Cold Case Squad, in part ?쐀ecause that?셲 where the money is. ??He added: ?쏻hen de Blasio came in, he talked about the tale of two cities. And then he?셲 done the complete opposite of what he said. It?셲 just business as usual. ??The Bronx?셲 struggles extend into prosecutions. In each of the last five years, prosecutors in the Bronx have declined to prosecute violent felony cases more than anywhere else in the city. And the rate of conviction in the Bronx is routinely the lowest in the city as well, but has ticked up this year to surpass Brooklyn?셲 rate through November as Bronx prosecutors work to streamline cases. Some cases have become even more difficult to win because of the   problem in the 40th Precinct, which has allowed defense lawyers to attack the credibility of officers who were implicated, said Patrice O?셎haughnessy, a spokeswoman for the Bronx District Attorney?셲 office. The district attorney, Darcel D. Clark, elected in 2015, said in a statement, ?쏧 was a judge here in the Bronx, and I heard from jurors that they can?셳 be impartial because they don?셳 trust the police. ??Against that tide of mistrust, Sergeant LoPuzzo?셲 detectives work 36 hours straight on some fresh cases. They buy Chinese takeout with their own money for a murder suspect. They carry surveillance videos home in hopes that their personal computers may enhance them better than a squad computer. They buy an urn for a homeless mother who has her murdered son?셲 ashes in a box. In the months after a killing, they can seem like the only people in this glittering city who are paying attention to the 40th Precinct?셲 homicide victims. Newly fatherless children go back to school without a therapist?셲 help. Victims??families wander confused through a courthouse and nearly miss an appearance. Newspapers largely ignore killings of people with criminal pasts, pushing them down the priority lists of the   chiefs at Police Headquarters. In a stuffy   squad room, the detectives of the 40th Precinct grapple with an inheritance of government neglect. They meet mothers who believe their sons might never have been murdered had a city guidance counselor listened to pleas to help them stay enrolled, or had a city housing worker fixed the locks or lights on a building. And the detectives work alongside a vicious system on the streets for punishing police cooperators. Young men scan court paperwork in prison, looking for the names of people who turned on them. One murder victim in the precinct this year was cast out of his crew after he avoided being arrested with them in a gang takedown some believed he was cooperating. A longtime 40th Precinct detective, Jeff Meenagh, said a witness in a homicide case was going to testify until he went back to his neighborhood and was told that anyone who testified would ?쐅et what you deserve. ??The allies Sergeant LoPuzzo makes are friendly only for so long. He helped clear a woman?셲 son of a robbery charge by locating surveillance video that proved he was not the robber. The mother started calling with tips under a code name  ??  about a gun under a car, for example. But she always refused to testify. And she cut ties this year after Sergeant LoPuzzo arrested her son in the stabbing of two people and her   in a shooting. New York City owns East 146th Street and the   buildings on each side. But James Fernandez, in the Betances Houses, said the reality on the ground was different: The drug boss ran the block. By October, Mr. Fernandez was increasingly afraid  ??  and fed up. Mr. Fernandez and his wife went so far as to give officers keys to the building door, so they could get in whenever they wanted, showed them the videos and offered them   access to his camera so they could see what was happening in the hallway. A couple of officers said they needed a supervisor?셲 permission to do more. Others answered that the young men were only making threats. Officers occasionally stopped outside their building, causing the young men to scatter, but did not come inside, Mr. Fernandez said. The menacing worsened. Mr. Fernandez?셲 daughter was harassed as she arrived home from school. She grew more and more distressed, and her parents had her start seeing a therapist. Mr. Fernandez made several complaints at the office of the borough president, Ruben Diaz Jr. and visited a victim?셲 advocate in the district attorney?셲 office. On Oct. 20, 2015, he sent an online note to the police commissioner?셲 office. ?쏻e went to all proper channels for help,??the note said. ?쏝oth precincts failed us, except 2 officers who helped us, but their hands are tied. No one else to turn to. I have months of video of multiple crimes taking place and we are in extreme danger. ????0th and PSA 7 won?셳 do anything,??he wrote, referring to the local substation. ?쏱lease we need to speak to some one with authority. ??The local substation commander, Deputy Inspector Jerry O?셎ullivan, and the Bronx narcotics unit were alerted to the complaints. But Mr. Fernandez said he never heard from them. So he relied on his own street instincts to protect his family. He made pleas to a man he thought was employing the dealers in the hallway. The activity quieted briefly, but it returned after the young men rented a room in a woman?셲 apartment upstairs. Mr. Fernandez approached a different man who he learned was the boss of the operation. The man agreed to ask the dealers to calm down. He even hired a drug customer to sweep the hallway, Mr. Fernandez said. But two weeks later, the dealing and the harassment resumed. So he went to his old Lower East Side bosses, who hired men to trail his wife and daughter on their way out of the building and make sure they made it safely to school. At other times they sat outside the Betances Houses. He also bought two bulletproof vests, for about $700 each. He could not find one small enough for his daughter. ?쏧 have no faith in the City of New York, I have no faith in the police, I have no faith in the politicians,??Mr. Fernandez said. ?쏷he only thing I know for sure: God, if we?셱e in a situation again, I will be left to defend my family. ??Paying such close attention to what was happening in the hallway, Mr. Fernandez said he learned some details about two recent homicides that the 40th Precinct was investigating. But because his calls for help were going nowhere, he said he decided not to put himself in greater risk by talking: He would not tell the police what he had learned. ?쏧?셫 bending over backward, and nobody?셲 not even doing anything,??he said. ?쏻hy am I going to help you, if you ain?셳 going to help me???By last January, a new neighborhood coordination officer was working with residents of the Betances Houses, and ended up with the most arrests in his housing command, Inspector O?셎ullivan said. Chief Boyce said that the silos in which gang and narcotics detectives used to work made responding to complaints more difficult, but that the recent restructuring would remove those obstacles. ?쏯o one should live like Mr. Fernandez lived, with people dealing drugs outside of his apartment,??he said. Mr. Fernandez?셲 complaints did not spur any arrests, but two men from the hallway were caught separately this year in shootings. One of them, whom Mr. Fernandez named in a police report, was charged this summer with hitting an officer with a metal folding chair and firing three gunshots into a crowd, court papers say. He is being held on Rikers Island on an attempted murder charge. That was too late for Mr. Fernandez. By May, he had moved his family away."
sentn = 10


def SKMN(input1, sentcount):
    summarizer = KeysentenceSummarizer(tokenize=komoran_tokenizer, similarity='textrank', min_sim=0.3)
    tr = TextRank()
    tr.loadSents(RawSentence(input1), lambda sent: filter(lambda x: x not in stop_words, preprocessing(input1)))
    tr.build()
    sents = sent_tokenize(input1)
    keysent1 = summarizer.summarize(sents, topk=sentcount)
    keysent2 = sent_tokenize(summarize(input1, ratio=0.1))
    keysent3 = sent_tokenize(tr.summarize(sentcount))
    keysent4 = summarizer.summarize(sents, topk=sentcount * 2)
    Mixlist = mixlist(mixlist(mixlist(keysent1, keysent2), keysent3), keysent4)
    mixlen = len(Mixlist)
    countlist = [0] * mixlen
    countlist = mcountlist(Mixlist, keysent1, countlist)
    countlist = mcountlist(Mixlist, keysent2, countlist)
    countlist = mcountlist(Mixlist, keysent3, countlist)
    countlist = mcountlist(Mixlist, keysent4, countlist)
    maxnum = max(countlist)
    rcountlist = [0] * mixlen
    for i in range(mixlen):
        rcountlist[i] = maxnum - countlist[i] + 1
    from queue import PriorityQueue
    que = PriorityQueue()
    for i in range(mixlen):
        que.put((rcountlist[i], Mixlist[i]))
    for i in range(sentcount):
        print(que.get()[1])

if __name__ == '__main__':
    SKMN(input1, sentn)
