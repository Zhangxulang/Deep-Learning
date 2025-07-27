# 作者: 王道 龙哥
# 2024年07月18日20时03分54秒
# dartou@qq.com
def parse_input():
    import sys
    input = sys.stdin.read
    data = input().split()

    idx = 0
    n = int(data[idx])
    idx += 1
    m = int(data[idx])
    idx += 1

    map_data = []
    for i in range(n):
        map_data.append(data[idx:idx + m])
        idx += m

    blue_bases = []
    red_bases = []

    num_blue_bases = int(data[idx])
    idx += 1
    for _ in range(num_blue_bases):
        x = int(data[idx])
        idx += 1
        y = int(data[idx])
        idx += 1
        fuel = int(data[idx])
        idx += 1
        missiles = int(data[idx])
        idx += 1
        defense = int(data[idx])
        idx += 1
        value = int(data[idx])
        idx += 1
        blue_bases.append((x, y, fuel, missiles, defense, value))

    num_red_bases = int(data[idx])
    idx += 1
    for _ in range(num_red_bases):
        x = int(data[idx])
        idx += 1
        y = int(data[idx])
        idx += 1
        fuel = int(data[idx])
        idx += 1
        missiles = int(data[idx])
        idx += 1
        defense = int(data[idx])
        idx += 1
        value = int(data[idx])
        idx += 1
        red_bases.append((x, y, fuel, missiles, defense, value))

    num_fighters = int(data[idx])
    idx += 1
    fighters = []
    for _ in range(num_fighters):
        x = int(data[idx])
        idx += 1
        y = int(data[idx])
        idx += 1
        fuel = int(data[idx])
        idx += 1
        missiles = int(data[idx])
        idx += 1
        fighters.append((x, y, fuel, missiles))

    return n, m, map_data, blue_bases, red_bases, fighters


def generate_commands():
    n, m, map_data, blue_bases, red_bases, fighters = parse_input()

    commands = []
    for i, (x, y, fuel, missiles) in enumerate(fighters):
        # Example move command: move fighter i to the right (dir = 3)
        commands.append(f"move {i} 3")

    return commands


def main():
    commands = generate_commands()
    for command in commands:
        print(command)
    print("OK")


if __name__ == "__main__":
    main()
