class Swarmllm < Formula
  desc "Decentralized peer-to-peer LLM inference network"
  homepage "https://github.com/enapt/SwarmLLM"
  version "0.1.0"
  license any_of: ["MIT", "Apache-2.0"]

  on_macos do
    on_arm do
      url "https://github.com/enapt/SwarmLLM/releases/download/v#{version}/swarmllm-macos-aarch64.tar.gz"
      # sha256 "UPDATE_WITH_ACTUAL_SHA256"
    end
  end

  on_linux do
    on_intel do
      url "https://github.com/enapt/SwarmLLM/releases/download/v#{version}/swarmllm-linux-x86_64.tar.gz"
      # sha256 "UPDATE_WITH_ACTUAL_SHA256"
    end
  end

  def install
    bin.install "swarmllm"
    etc.install "default.toml" => "swarmllm/default.toml" if File.exist?("default.toml")
  end

  def post_install
    (var/"swarmllm").mkpath
  end

  def caveats
    <<~EOS
      Data directory: #{var}/swarmllm
      Default config: #{etc}/swarmllm/default.toml

      Start the daemon:
        swarmllm run

      Open the dashboard:
        open http://localhost:8800
    EOS
  end

  service do
    run [opt_bin/"swarmllm", "run", "--data-dir", var/"swarmllm"]
    keep_alive true
    log_path var/"log/swarmllm.log"
    error_log_path var/"log/swarmllm.log"
  end

  test do
    assert_match "swarmllm", shell_output("#{bin}/swarmllm --version")
  end
end
