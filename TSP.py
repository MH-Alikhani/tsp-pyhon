import random, numpy, math, copy, matplotlib.pyplot as plt

noc = int(input("Eneter a valid number of cities:"))

random.seed(123)
cities = [random.sample(range(100), 2) for x in range(noc)]

tour = random.sample(range(noc), noc)

for temperature in numpy.logspace(0, 5, num=100000)[::-1]:
    [i, j] = sorted(random.sample(range(noc), 2))
    newTour = (
        tour[:i] + tour[j : j + 1] + tour[i + 1 : j] + tour[i : i + 1] + tour[j + 1 :]
    )
    if (
        math.exp(
            (
                sum(
                    [
                        math.sqrt(
                            sum(
                                [
                                    (
                                        cities[tour[(k + 1) % noc]][d]
                                        - cities[tour[k % noc]][d]
                                    )
                                    ** 2
                                    for d in [0, 1]
                                ]
                            )
                        )
                        for k in [j, j - 1, i, i - 1]
                    ]
                )
                - sum(
                    [
                        math.sqrt(
                            sum(
                                [
                                    (
                                        cities[newTour[(k + 1) % noc]][d]
                                        - cities[newTour[k % noc]][d]
                                    )
                                    ** 2
                                    for d in [0, 1]
                                ]
                            )
                        )
                        for k in [j, j - 1, i, i - 1]
                    ]
                )
            )
            / temperature
        )
        > random.random()
    ):
        tour = copy.copy(newTour)

print([[cities[tour[i % noc]][0], cities[tour[i % noc]][1]] for i in range(noc + 1)])

plt.plot(
    [cities[tour[i % noc]][0] for i in range(noc + 1)],
    [cities[tour[i % noc]][1] for i in range(noc + 1)],
    "xb-",
)
plt.show()


# This project was coded and designed by Mohammad Hosein Alikhani
