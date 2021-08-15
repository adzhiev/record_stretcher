<h1> Algorithm to stretch or compress 1-channel .wav record in duration</h1>
<p> The algorithm is implemented according to http://www.guitarpitchshifter.com/algorithm.html without the <em>Resample</em> part </p>
<p> Firstly, install the dependencies from <em>requirements.txt</em> </p> 
<p> To execute the prog run <em>main.py</em>:</p>
<p> <strong>python main.py</strong> </p>
<p> With default args it will produce 2 additional wav-records in <em>samples</em> directory that are twice longer and shorter in duration </p>
<p> You can make other stretched records using appropriate args. For example, if you want to make 3 wav records <em>half</em>, <em>twice</em> 
and <em>3 times</em> longer than your chosen record that is located in <em>path</em> with window size <em>1024</em>, then run: </p>
<p> <strong>python main.py -p path -sr 0.5 2 3 -ws 1024</strong> </p>
<p> The generated records will be in the same directory as <em>path</em>, but with additional prefixes according to the passed stretched ratios </p>
