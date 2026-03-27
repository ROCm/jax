mkdir -p ./xla_dumps/svg
for f in ./xla_dumps/*.dot; do
  [ -e "$f" ] || continue
  dot -Tsvg "$f" > "./xla_dumps/svg/$(basename "$f").svg"
done
