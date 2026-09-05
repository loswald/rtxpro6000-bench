ls -la /workspace/bench/patch_ple.py /workspace/bench/chain_c5.sh /workspace/bench/qwen38fn_c3.txt /workspace/bench/lists/qwen38fn_c3.txt 2>&1 | awk '{print $5, $9}'
mv -f /workspace/bench/qwen38fn_c3.txt /workspace/bench/lists/qwen38fn_c3.txt 2>/dev/null; chmod +x /workspace/bench/chain_c5.sh
python3 -c "import ast; ast.parse(open('/workspace/bench/patch_ple.py').read()); print('patch_ple parses')"
tmux has-session -t =chainc5 2>/dev/null && echo "chainc5 already running" || tmux new-session -d -s chainc5 "bash /workspace/bench/chain_c5.sh >> /workspace/results/chain_c.log 2>&1"
sleep 10; tmux ls | cut -d: -f1 | paste -sd' '; tail -4 /workspace/results/chain_c.log | cut -c1-160
