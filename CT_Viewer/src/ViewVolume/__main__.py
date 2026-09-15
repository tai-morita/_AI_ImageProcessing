from __future__ import annotations

from .viewer import view_volume


def main() -> None:
    ## @brief Start the viewer; data is selected with the LoadData button.
    coordinate, input_path = view_volume()
    print(f"Registered coordinate (x, y, z): {coordinate}")
    print(f"Input path: {input_path}")


if __name__ == "__main__":
    main()
