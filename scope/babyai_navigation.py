"""Remember only BabyAI's public egocentric observations and dispatched movement."""
from collections import deque
import re


DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))
PASSABLE = {"empty", "floor", "goal"}
OBJECTS = {"box", "ball", "key", "door"}
TARGET = re.compile(r"\b(?:(red|green|blue|purple|yellow|grey) )?(key|ball|box|door|object)\b")


def ahead(position, heading):
    x, y = position
    dx, dy = DIRS[heading]
    return x + dx, y + dy


def passable(tile):
    return bool(tile and (tile[0] in PASSABLE or (tile[0] == "door" and tile[2] == 0)))


class PublicLocalMap:
    def __init__(self):
        self.pose = (0, 0)
        self.heading = 0
        self.tiles = {}
        self.seen = set()
        self.uncertain = False
        self.frame = 0

    @staticmethod
    def project(image, heading, pose, objects, colors):
        width, depth, _ = image.shape
        dx, dy = DIRS[heading]
        x0, y0 = pose
        tiles = {}
        for col in range(width):
            for row in range(depth):
                if (col, row) == (width // 2, depth - 1):
                    continue
                raw_kind, raw_color, state = map(int, image[col, row])
                kind = objects[raw_kind]
                if kind == "unseen":
                    continue
                front, right = depth - 1 - row, col - width // 2
                tiles[(x0 + front * dx - right * dy, y0 + front * dy + right * dx)] = (kind, colors[raw_color], state)
        return tiles

    def observe(self, image, direction, command, objects, colors):
        previous = self.pose
        if command == "forward" and passable(self.tiles.get(ahead(previous, self.heading))):
            self.pose = ahead(previous, self.heading)
        elif command == "forward" and ahead(previous, self.heading) not in self.tiles:
            # Unobserved forward result cannot establish the absolute location.
            self.uncertain = True
            self.frame += 1
            self.pose, self.tiles, self.seen = (0, 0), {}, set()
        self.heading = int(direction)
        visible = self.project(image, self.heading, self.pose, objects, colors)
        def conflicts(projected):
            return sum((tile[0] == "wall") != (self.tiles[pos][0] == "wall")
                       for pos, tile in projected.items() if pos in self.tiles)
        if conflicts(visible):
            self.uncertain = True
            self.frame += 1
            self.pose, self.tiles, self.seen = (0, 0), {}, set()
            visible = self.project(image, self.heading, self.pose, objects, colors)
        self.tiles.update(visible)
        self.seen.add(self.pose)
        return visible

    def next_steps(self, targets):
        """Find a public route to face the target, or explore an observed frontier."""
        queue = deque([(self.pose, self.heading, ())])
        visited = {(self.pose, self.heading)}
        frontier = None
        while queue:
            position, heading, path = queue.popleft()
            front = ahead(position, heading)
            if front in targets:
                return path, True
            if path and front not in self.tiles and frontier is None:
                frontier = path
            transitions = [(position, (heading - 1) % 4, "left"), (position, (heading + 1) % 4, "right")]
            if passable(self.tiles.get(front)) or (front not in self.tiles and front in self.seen):
                transitions.append((front, heading, "forward"))
            for new_position, new_heading, action in transitions:
                if (new_position, new_heading) not in visited:
                    visited.add((new_position, new_heading))
                    queue.append((new_position, new_heading, (*path, action)))
        return frontier or (), False

    def nearest_reachable(self, targets):
        routes = []
        for pos in targets:
            path, reached = self.next_steps({pos})
            if reached:
                routes.append((len(path), pos))
        return min(routes)[1] if routes else None


def mission_stages(mission):
    ordered = re.split(r"\s*,?\s+(?:then|and then)\s+", mission.lower())
    return [clause for clause in ordered if TARGET.search(clause)]


def matching_targets(tiles, instruction):
    match = TARGET.search(instruction)
    if not match:
        return set()
    color, kind = match.groups()
    return {pos for pos, tile in tiles.items()
            if (not color or tile[1] == color) and (tile[0] == kind or kind == "object" and tile[0] in OBJECTS)}


def ordered_mission(mission):
    clauses = re.split(r"\s*,?\s+(and then|then|after you|before you|after|before|and)\s+", mission.lower())
    if len(clauses) == 3 and clauses[1] in {"after", "after you"}:
        clauses[0], clauses[2] = clauses[2], clauses[0]
    stages = []
    for clause in clauses[::2]:
        clause = clause.strip(" ,.").removeprefix("and ")
        if not TARGET.search(clause):
            continue
        for verb, action in (("pick up ", "pickup"), ("go to ", "face"), ("open ", "toggle"), ("put ", "deliver")):
            if not clause.startswith(verb):
                continue
            value = clause[len(verb):]
            if action == "deliver":
                obj, separator, target = value.partition(" next to ")
                if separator and TARGET.search(target):
                    stages.append((action, obj, target))
            else:
                stages.append((action, value, None))
            break
    return stages


def matches_item(held, description):
    if not held:
        return False
    color, kind = held.split(" ", 1)
    return bool(matching_targets({(0, 0): (kind, color, 0)}, description))


def delivery_tiles(tiles, targets):
    return {p for pos in targets for heading in range(4) if (p := ahead(pos, heading)) in tiles
            and tiles[p][0] in {"empty", "floor"}}
