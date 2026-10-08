# FieldOps Work Order Management - Software Requirements Specification

Version 0.9 (draft). Fictional sample document written for the ai-requirements-agent demo.

## 1 Introduction

FieldOps is a work order management system for a regional HVAC and electrical maintenance company. Dispatchers in the operations centre receive service requests from customers, create work orders and assign them to field technicians who travel to customer sites. Today this is coordinated with spreadsheets and phone calls, which causes double bookings, missed SLAs and lost paperwork.

### 1.1 Scope

FieldOps covers intake of service requests, scheduling and dispatch, the technician mobile app, parts usage, customer notifications and operational reporting. Payroll and invoicing remain in the existing ERP and are out of scope for this release.

### 1.2 Actors

- Dispatcher: creates, schedules and assigns work orders.
- Field technician: performs the work on site using the mobile app.
- Customer: requests service and receives status updates.
- Operations manager: monitors SLAs and team performance.

## 2 Functional Requirements

### 2.1 Work order intake

- The system shall allow dispatchers to create work orders from customer service requests.
- Each work order has a reference number, customer, site address, priority (low, normal, high, emergency), status (new, scheduled, in progress, on hold, completed, cancelled), problem description, scheduled date and assigned technician.
- The system shall validate that the site address belongs to an existing customer before a work order is saved.
- Dispatchers must be able to update work orders until they are completed.
- Dispatchers can cancel work orders when the customer withdraws the request, and the system shall record the cancellation reason.

### 2.2 Scheduling and dispatch

- The system shall allow dispatchers to assign work orders to technicians based on skills and availability.
- Each technician has a name, email, phone number, skills, home depot and working hours.
- Dispatchers shall be able to reassign a work order when the assigned technician is unavailable.
- The system should suggest the nearest available technician with the required skill for emergency work orders.
- Operations managers can view the dispatch board showing all scheduled work orders for the day.

### 2.3 Technician mobile app

- Technicians must be able to view their assigned work orders on the mobile app.
- Technicians shall be able to complete work orders on site and capture the customer signature.
- Technicians can record parts used on a work order, and each work order can have multiple parts used.
- Each part used has a part number, description, quantity and unit cost.
- The mobile app must work offline and synchronise changes within 2 minutes of connectivity being restored.

### 2.4 Customer notifications

- When a work order is scheduled, the system shall send the customer an email with the appointment window.
- As a customer, I want to receive a notification when the technician is on the way so that I can make sure someone is at the site.
- Given a work order is in progress, when the technician marks it as en route, then the customer receives an SMS with the estimated arrival time.

### 2.5 Reporting

- Operations managers must be able to export a monthly SLA report.
- The system shall track first-time fix rate per technician.

## 3 Non-Functional Requirements

### 3.1 Performance

- The dispatch board shall load within 2 seconds for up to 500 open work orders.
- Search results should be returned in under 500 ms at the 95th percentile.

### 3.2 Availability and reliability

- The system must be available 99.5% of the time during business hours (07:00-19:00 local time).
- Nightly backups shall be retained for 35 days and a restore must be tested every quarter.

### 3.3 Security

- All users must authenticate via the company's single sign-on provider.
- Customer personal data must be encrypted at rest and in transit using TLS 1.2 or higher.
- The system shall keep an audit trail of every status change on a work order.

### 3.4 Scalability and usability

- The platform should support 200 concurrent technicians without degradation.
- The mobile app should be easy to use with gloves on.

## 4 Open Issues

- Should emergency work orders bypass the normal approval of overtime?
- Integration with the ERP for parts stock levels is to be confirmed.
