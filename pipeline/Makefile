# Zero-dependency alternative to flow.py. Run from the repo root:  make -C pipeline <target>
REGION ?= eu-north-1

.PHONY: init plan up ingest catalog curate dq status down destroy-all fmt

init:      ; cd infra && terraform init -input=false
plan:      ; cd infra && terraform init -input=false && terraform plan -out tf.plan
up:        ; cd infra && terraform apply -input=false tf.plan
fmt:       ; cd infra && terraform fmt

ingest:    ; python flow.py run --step ingest  --region $(REGION)
catalog:   ; python flow.py run --step catalog --region $(REGION)
curate:    ; python flow.py run --step curate  --region $(REGION)
dq:        ; python flow.py run --step dq      --region $(REGION)
status:    ; python flow.py status             --region $(REGION)

# tear down compute only; the bucket + catalog + curated tier stay
down:      ; python flow.py down

# tear down EVERYTHING including the data (asks twice)
destroy-all: ; cd infra && terraform destroy -input=false
