# Laplacian text generator

This is an old-school text generator, producing text roughly on the same order of coherence as Markov-chain based generators. It's just a little more sophisticated, as it's based on graph potentials (i.e. solutions of the Laplace-Poisson PDE).

Right now a text base is hardwired into it. I could have spent fifteen minutes more adding a "upload custom edge" widget, but it really requires long text to work well; it probably depends on the variety of unusual words and neologism its base texts are using. Some of them are currently in print, but I claim fair use / educational and research purposes. Furthermore: to the best of my knowledge, the deceased author would probably have given this a good belly laugh, followed by a hard lifelong smoker cough.

(This means however that I can't give the repo any simple open source license. Here's an USE AT YOUR OWN RISK disclaimer: it took me about 40 minutes to make, and there are certainly bugs.)