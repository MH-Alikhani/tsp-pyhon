<h1>Traveling Salesman Problem (TSP) with Simulated Annealing</h1>

<p>This project solves the Traveling Salesman Problem (TSP) using simulated annealing, a probabilistic technique for approximating the global minimum of a given function.</p>

<h2>Installation</h2>

<p>To run this code, you will need Python 3 and the following libraries:</p>
<ul>
    <li><code>random</code></li>
    <li><code>numpy</code></li>
    <li><code>matplotlib</code></li>
</ul>

<p>You can install these libraries using pip:</p>

<pre>pip install random numpy matplotlib</pre>

<h2>Usage</h2>

<p>To run the code, follow these steps:</p>

<ol>
    <li>Open a terminal window and navigate to the directory containing the <code>TSP.py</code> file.</li>
    <li>Run the following command:</li>

<pre>python TSP.py</pre>

<p>This will prompt you to enter the number of cities in the TSP problem. Enter the number and press enter. The code will then solve the TSP and plot the optimal tour.</p>
</ol>

<h2>Code Refactored</h2>

<p>The code has been refactored to improve its readability and maintainability. This includes:</p>
<ul>
    <li>Using functions to modularize the code</li>
    <li>Improving variable naming</li>
    <li>Replacing <code>sum()</code> with a nested loop</li>
    <li>Simplifying the logic for accepting or rejecting a new solution</li>
</ul>

<h2>Performance</h2>

<p>The refactored code should have similar performance to the original code. The annealing loop will typically terminate after a few thousand iterations. The optimal tour will be displayed on the screen.</p>
