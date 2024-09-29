class AIscientistGraph:
    def __init__(self, name, research_field):
        self.name = name
        self.research_field = research_field

    def __str__(self):
        return f"{self.name} is an AI scientist working on {self.research_field}"

    def __repr__(self):
        return f"AIscientist({self.name}, {self.research_field})"


if __name__ == "__main__":
    image = ""
    with open("../data/research_architecture.png", "wb") as f:
        f.write(image.data)
