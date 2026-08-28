# Privacy Policy

**Project:** LabMiM Micrometeorology & Solar Radiation Intelligence
**Maintainer:** Micrometeorology Laboratory (LabMiM), Universidade Federal da Bahia (UFBA)
**Repository:** https://github.com/Bruno-Mascarenhas/micrometeorology
**Last updated:** 2026-08-27

## 1. What this project is

This repository contains open-source scientific software for atmospheric
research: numerical weather model post-processing, meteorological sensor
ingestion, and machine-learning pipelines for solar irradiance estimation. It is
distributed as source code under the MIT License and is run by each user on
their own computing infrastructure.

There is no hosted service, no website that collects visitor data, and no
telemetry. The maintainers operate no server that receives data from
installations of this software.

## 2. Data the software processes

Run locally, the software reads and writes:

- meteorological measurements recorded by the LabMiM weather station,
- all-sky camera imagery and derived video frames,
- numerical weather prediction model output,
- derived datasets, model checkpoints and evaluation artifacts.

These are physical measurements of the atmosphere. They are not personal data
and do not identify individuals.

## 3. Google account access

The software can optionally synchronise research datasets with a user's own
Google Drive, using [rclone](https://rclone.org/) as the transfer client. When a
user chooses to enable this:

- Google Drive authorisation is granted by the user, through Google's own OAuth
  consent flow, to the user's own account.
- The resulting access and refresh tokens are stored **only** on the user's
  machine, in that user's rclone configuration file. They are never transmitted
  to the maintainers and never committed to this repository.
- The requested access is used exclusively to read and write the research files
  the user selects, in the folders the user designates.
- No file content, file listing, account identifier or email address is sent to
  the maintainers or to any third party.
- Authorisation can be revoked at any time by the user, from
  https://myaccount.google.com/permissions, or by deleting the local rclone
  configuration.

The maintainers have no access to any user's Google account or Google Drive
contents.

## 4. Data sharing

No data processed by this software is sold, rented, or shared with third
parties. The maintainers do not receive it.

## 5. Retention

Because no data is collected by the maintainers, none is retained by them. Data
produced by a local installation remains under the sole control of whoever runs
that installation, and is deleted by deleting those local files.

## 6. Third-party services

Users who enable Google Drive synchronisation are additionally subject to
Google's own terms and privacy policy: https://policies.google.com/privacy.
Users who obtain the software from GitHub are subject to GitHub's policies.

## 7. Your rights

Users in Brazil are protected by the Lei Geral de Proteção de Dados (LGPD, Law
13.709/2018), and users in the European Economic Area by the GDPR. Because this
project does not collect personal data, there is normally nothing for the
maintainers to disclose, correct, or erase. Requests may nonetheless be sent
through the contact channel below and will be answered.

## 8. Changes

Revisions to this policy are published in this file, in the repository's public
history, with the date above updated.

## 9. Contact

Questions about this policy can be raised at
https://github.com/Bruno-Mascarenhas/micrometeorology/issues, or addressed to
the Micrometeorology Laboratory (LabMiM), Instituto de Física, Universidade
Federal da Bahia, Salvador, Bahia, Brazil.
