{
  description = "Gold Machine - F# quantitative strategy lab";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python3.withPackages (ps: with ps; [ pandas requests beautifulsoup4 lxml ]);
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [ dotnet-sdk_10 python ];

          shellHook = ''
            export LD_LIBRARY_PATH="$(find /nix/store -maxdepth 3 -path '*/libstdc++.so.6' -exec dirname {} \; 2>/dev/null | head -1):$LD_LIBRARY_PATH"
            export DOTNET_ROOT="$(dotnet --list-runtimes 2>/dev/null | grep Microsoft.NETCore.App | head -1 | awk '{print $3}' | tr -d '[]' | xargs dirname | xargs dirname)"

            if [ ! -d .venv ]; then
              python -m venv .venv --system-site-packages
              .venv/bin/pip install -q akshare jupyterlab
            fi
            source .venv/bin/activate

            echo "=== Gold Machine ==="
            echo "dotnet $(dotnet --version)"
            echo "python $(python --version)"
            python -c "import akshare; print(f'akshare: {akshare.__version__}')" 2>/dev/null || \
              echo "akshare: not available"
          '';
        };
      });
}
